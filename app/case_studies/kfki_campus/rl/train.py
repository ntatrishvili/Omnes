"""Training of the agents on the KFKI environment: set-up, warm start from a rule, PPO.

Thin orchestration on top of ``KfkiChargingEnv``, ``RLAgent`` and ``evaluate``; the case study
is run by ``run.py``.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.excel_io import CampusData
from app.case_studies.kfki_campus.rl.baselines import BASELINES
from app.case_studies.kfki_campus.rl.env import KfkiChargingEnv
from app.case_studies.kfki_campus.rl.evaluate import evaluate_policies
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


def eval_day_list(env: KfkiChargingEnv, n: Optional[int]) -> list[int]:
    """``n`` test days spread over the PV range (sunny and dull days alike), sorted by day.

    Days drawn at fixed intervals can miss the sunny days on which the PV surplus, and so
    the policies, differ; ranking by daily PV energy and taking evenly spaced ranks covers them.
    """
    days = list(env.test_days)
    if not n or n >= len(days):
        return days
    spd = env.steps_per_day
    pv = np.array([env._pv_total[d * spd : (d + 1) * spd].sum() for d in days])
    ranks = np.linspace(0, len(days) - 1, n).round().astype(int)
    return sorted(days[i] for i in np.argsort(pv)[ranks])


RULES = ("uncontrolled", "pv_following", "line_following", "least_laxity", "grid_guard")


def pick_rule(problem: RLProblem, setup: Setup, n_days: int = 8) -> str:
    """Name of the rule-based policy with the best reward of the problem on training days."""
    agents = {
        k: v for k, v in reference_agents(problem, oracle=False).items() if k in RULES
    }
    env = make_env(problem, setup, "train")
    days = env.train_days[:: max(1, len(env.train_days) // n_days)][:n_days]
    k = evaluate_policies(env, agents, days, seed=5, keep_episodes=False).kpis
    return str(k.groupby("policy")["reward"].mean().idxmax())


def collect_demos(
    problem: RLProblem, setup: Setup, rule: RLAgent, n_days: int = 250, seed: int = 7
):
    """(observations, actions, discounted returns) of a rule on random training days.

    The rule gives the charger actions; with an agent-controlled battery the battery setpoint
    it actually used (its own rule) is appended, so the policy can clone the whole behaviour.
    """
    env = make_env(problem, setup, "train")
    rng = np.random.default_rng(seed)
    gamma = problem.agent.gamma
    days = rng.choice(
        env.train_days, size=min(n_days, len(env.train_days)), replace=False
    )
    obs_l, act_l, ret_l = [], [], []
    for i, day in enumerate(days):
        rule.bind(env)
        obs, _ = env.reset(seed=seed + i, options={"day": int(day)})
        o, a, r = [], [], []
        done = False
        while not done:
            act = np.clip(np.asarray(rule.predict(obs), float).reshape(-1), 0.0, 1.0)
            o.append(obs)
            obs, reward, _, done, _ = env.step(act)
            if problem.battery_policy == "agent" and len(act) == len(env.chargers):
                act = np.append(act, env._bat_action)  # what the battery rule did
            a.append(act)
            r.append(reward)
        g, ret = 0.0, np.zeros(len(r))
        for t in range(len(r) - 1, -1, -1):
            g = r[t] + gamma * g
            ret[t] = g
        obs_l += o
        act_l += a
        ret_l += list(ret)
    return np.array(obs_l, np.float32), np.array(act_l), np.array(ret_l)


def train_agent(
    problem: RLProblem,
    setup: Setup,
    out_dir: Path,
    seed: int = 0,
    total_timesteps: Optional[int] = None,
    eval_every: int = 5000,
    eval_days: int = 5,
    name: Optional[str] = None,
    warm_start: bool = False,
) -> tuple[RLAgent, pd.DataFrame]:
    """Train one agent; returns it with its learning curve (and saves both under ``out_dir``).

    With ``warm_start`` the best rule of the problem (``pick_rule``) is cloned into the
    policy and the critic before PPO starts (smaller initial action noise), so the learning
    curve starts at the rule's level and PPO refines it.
    """
    out_dir = Path(out_dir)
    name = name or f"{problem.agent.algo.lower()}_seed{seed}"
    problem = copy.deepcopy(problem)
    problem.agent.seed = seed
    if warm_start:
        problem.agent.log_std_init = -1.0
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
    days = eval_day_list(eval_env, eval_days)
    ref = evaluate_policies(
        eval_env,
        {"uncontrolled": BASELINES["uncontrolled"](problem)},
        days,
        seed=1000,
        keep_episodes=False,
    ).kpis["reward"].mean()  # the curve shows the gain over uncontrolled charging
    if warm_start:
        rule_name = pick_rule(problem, setup)
        obs, act, ret = collect_demos(
            problem,
            setup,
            reference_agents(problem, oracle=False)[rule_name],
            seed=seed,
        )
        agent.prepare(venv)
        info = agent.pretrain(obs, act, ret, epochs=150, seed=seed)
        log.info(f"warm start from '{rule_name}': {info}")
    callback, rows = _curve_callback(
        agent, eval_env, days, eval_every, name, seed, float(ref)
    )
    agent.train(venv, total_timesteps, callback=callback)
    callback.evaluate_now()  # final point
    curve = pd.DataFrame(rows)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    agent.save(out_dir / "models" / name)
    curve.to_csv(out_dir / "models" / f"{name}_curve.csv", index=False)
    return agent, curve


def _curve_callback(agent, eval_env, days, every, name, seed, ref_reward=0.0):
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
                    reward_vs_uncontrolled=float(k["reward"].mean() - ref_reward),
                    SC=float(k["SC"].mean()),
                    SS=float(k["SS"].mean()),
                    ev_unmet_kwh=float(k["ev_unmet_kwh"].mean()),
                    ev_pv_share=float(k["ev_pv_share"].mean()),
                    line_overload_kvah=float(k["line_overload_kvah"].mean()),
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
        "line_following": BASELINES["line_following"](problem),
        "least_laxity": BASELINES["least_laxity"](problem),
        "random": RandomAgent(problem, seed=123),
    }
    if problem.battery_policy == "agent":  # the rules that also dispatch the battery
        agents["grid_guard"] = BASELINES["grid_guard"](problem)
    if oracle:
        agents["oracle"] = BASELINES["oracle"](problem)
    return agents
