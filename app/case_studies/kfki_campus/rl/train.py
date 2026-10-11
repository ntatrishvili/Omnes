"""Training, evaluation and experiments (reward sweep, event-source generalisation).

Thin orchestration on top of ``KfkiChargingEnv``, ``RLAgent`` and ``evaluate``; the command
line is ``run_rl.py``.
"""

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.excel_io import CampusData
from app.case_studies.kfki_campus.rl.baselines import BASELINES
from app.case_studies.kfki_campus.rl.env import KfkiChargingEnv, charger_ratings
from app.case_studies.kfki_campus.rl.evaluate import (
    Evaluation,
    evaluate_policies,
    full_period_run,
)
from app.case_studies.kfki_campus.rl.sessions import read_sessions
from app.infra.logging_setup import get_logger
from app.operation.rl.agent import BasicPPOAgent, RandomAgent, RLAgent, make_agent
from app.operation.rl.problem import RLProblem

log = get_logger(__name__)


@dataclass
class Setup:
    """What the environments are built from (shared by train / test envs).

    :ivar Path csv: Time-series CSV
    :ivar CampusData data: Input workbooks
    :ivar str time_start: First time stamp
    :ivar str time_end: End (exclusive)
    :ivar list events: Deterministic charging events (or ``None``)
    """

    csv: Path
    data: CampusData
    time_start: str
    time_end: str
    events: Optional[list] = None
    resolution: Optional[str] = None


def load_events(path: Path, problem: RLProblem, setup: Setup) -> list:
    """Read the session file of the problem for the chargers and period of the setup."""
    # all chargers of the workbook are valid ids; the environment uses the first n_chargers
    chargers = charger_ratings(
        setup.data, len(setup.data.ev_chargers), problem.action.charger_kw
    )
    period = (pd.Timestamp(setup.time_start), pd.Timestamp(setup.time_end))
    events, report = read_sessions(Path(path), chargers, period)
    log.info(
        f"{len(events)} sessions read from {path} ({len(report)} adjusted / dropped)"
    )
    return events


def make_env(problem: RLProblem, setup: Setup, split: str = "train") -> KfkiChargingEnv:
    return KfkiChargingEnv(
        problem,
        setup.csv,
        setup.data,
        setup.time_start,
        setup.time_end,
        setup.events,
        split,
        setup.resolution,
    )


def with_events(problem: RLProblem, **changes) -> RLProblem:
    """Copy of the problem with other ``events`` settings (e.g. ``source="file"``)."""
    p = copy.deepcopy(problem)
    for k, v in changes.items():
        setattr(p.events, k, v)
    return p


def train_agent(
    problem: RLProblem,
    setup: Setup,
    out_dir: Path,
    seed: int = 0,
    total_timesteps: Optional[int] = None,
    eval_every: int = 5000,
    eval_days: int = 5,
    name: Optional[str] = None,
) -> tuple[RLAgent, pd.DataFrame]:
    """Train one agent; returns it with its learning curve (and saves both under ``out_dir``)."""
    out_dir = Path(out_dir)
    name = name or f"{problem.agent.algo.lower()}_seed{seed}"
    problem = copy.deepcopy(problem)
    problem.agent.seed = seed
    agent = make_agent(problem)
    if not isinstance(agent, BasicPPOAgent):  # non-learning agent: nothing to train
        return agent, pd.DataFrame()
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    n_envs = problem.agent.n_envs
    venv = DummyVecEnv(
        [lambda: Monitor(make_env(problem, setup, "train")) for _ in range(n_envs)]
    )
    eval_env = make_env(problem, setup, "test")
    days = eval_env.test_days[:: max(1, len(eval_env.test_days) // eval_days)][
        :eval_days
    ]
    callback, rows = _curve_callback(agent, eval_env, days, eval_every, name, seed)
    agent.train(venv, total_timesteps, callback=callback)
    callback.evaluate_now()  # final point
    curve = pd.DataFrame(rows)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    agent.save(out_dir / "models" / name)
    curve.to_csv(out_dir / "models" / f"{name}_curve.csv", index=False)
    return agent, curve


def _curve_callback(agent, eval_env, days, every, name, seed):
    from stable_baselines3.common.callbacks import BaseCallback

    rows: list = []

    class Curve(BaseCallback):
        def __init__(self):
            super().__init__()
            self._next = 0

        def evaluate_now(self):
            agent.model = self.model if self.model is not None else agent.model
            ev = evaluate_policies(
                eval_env, {name: agent}, days, seed=1000, keep_episodes=False
            )
            k = ev.kpis
            buf = self.model.ep_info_buffer
            rows.append(
                dict(
                    run=name,
                    seed=seed,
                    timesteps=int(self.num_timesteps),
                    eval_reward=float(k["reward"].mean()),
                    SC=float(k["SC"].mean()),
                    SS=float(k["SS"].mean()),
                    ev_unmet_kwh=float(k["ev_unmet_kwh"].mean()),
                    train_reward=(
                        float(np.mean([e["r"] for e in buf])) if len(buf) else np.nan
                    ),
                )
            )

        def _on_training_start(self):
            self.evaluate_now()
            self._next = every

        def _on_step(self):
            if self.num_timesteps >= self._next:
                self.evaluate_now()
                self._next += every
            return True

    return Curve(), rows


def reference_agents(problem: RLProblem, oracle: bool = True) -> dict[str, RLAgent]:
    """Baseline policies compared with the learned agent."""
    agents: dict[str, RLAgent] = {
        "uncontrolled": BASELINES["uncontrolled"](problem),
        "pv_following": BASELINES["pv_following"](problem),
        "random": RandomAgent(problem, seed=123),
    }
    if oracle:
        agents["oracle"] = BASELINES["oracle"](problem)
    return agents


def evaluate_all(
    problem: RLProblem,
    setup: Setup,
    agents: dict[str, RLAgent],
    out_dir: Path,
    days: Optional[Sequence[int]] = None,
    full_period: bool = True,
    seed: int = 0,
) -> Evaluation:
    """Evaluate on the test days (and one full-period run each); write the result files."""
    out_dir = Path(out_dir)
    env = make_env(problem, setup, "test")
    ev = evaluate_policies(env, agents, days, seed)
    ev.kpis.to_csv(out_dir / "rl_kpis.csv", index=False)
    with open(out_dir / "episodes.pkl", "wb") as f:
        pickle.dump(ev.episodes, f)
    if full_period:
        rows = [full_period_run(env, a, n, out_dir, seed) for n, a in agents.items()]
        pd.DataFrame(rows).to_csv(out_dir / "rl_full_period.csv", index=False)
    return ev


def summarize(kpis: pd.DataFrame) -> pd.DataFrame:
    """Mean per policy of the main KPIs (rows: policies)."""
    cols = [
        "reward",
        "SC",
        "SS",
        "import_kwh",
        "export_kwh",
        "ev_unmet_kwh",
        "ev_satisfaction",
        "ev_pv_share",
        "ev_peak_kw",
        "battery_cycles",
        "v_max_pu",
        "viol_frac",
        "trafo_max_loading_pct",
        "line_max_loading_pct",
        "losses_kwh",
    ]
    cols = [c for c in cols if c in kpis]
    return kpis.groupby("policy", sort=False)[cols].mean()


def sweep_weight(
    problem: RLProblem,
    setup: Setup,
    out_dir: Path,
    metric: str,
    values: Sequence[float],
    timesteps: Optional[int] = None,
    days: int = 10,
) -> pd.DataFrame:
    """Train one agent per weight of ``metric`` and evaluate it (rows: weight)."""
    rows = []
    for w in values:
        p = copy.deepcopy(problem)
        for t in p.reward:
            if t.metric == metric:
                t.weight = float(w)
        name = f"{metric}_w{w:g}"
        agent, _ = train_agent(
            p, setup, Path(out_dir) / "sweep", 0, timesteps, name=name
        )
        env = make_env(p, setup, "test")
        test = env.test_days[:: max(1, len(env.test_days) // days)][:days]
        k = evaluate_policies(env, {name: agent}, test, 0, keep_episodes=False).kpis
        rows.append(
            {
                "metric": metric,
                "weight": w,
                **k.drop(columns=["policy"]).mean(numeric_only=True).to_dict(),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "rl_sweep.csv", index=False)
    return df


def generalization(
    problem: RLProblem,
    setup: Setup,
    out_dir: Path,
    timesteps: Optional[int] = None,
    days: int = 10,
) -> pd.DataFrame:
    """Train on each event source (file / generator / mixed), test on file and generator events."""
    if setup.events is None:
        raise ValueError("generalisation needs deterministic events (session file)")
    rows = []
    for train_src in ("file", "generator", "mixed"):
        p = with_events(problem, source=train_src)
        agent, _ = train_agent(
            p,
            setup,
            Path(out_dir) / "generalization",
            0,
            timesteps,
            name=f"train_{train_src}",
        )
        for test_src in ("file", "generator"):
            pt = with_events(problem, source=test_src)
            env = make_env(pt, setup, "test")
            test = env.test_days[:: max(1, len(env.test_days) // days)][:days]
            k = evaluate_policies(
                env, {"agent": agent}, test, 0, keep_episodes=False
            ).kpis
            rows.append(
                {
                    "train_source": train_src,
                    "test_source": test_src,
                    **k.drop(columns=["policy"]).mean(numeric_only=True).to_dict(),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "rl_generalization.csv", index=False)
    return df
