"""Evaluation of policies on the KFKI environment: KPI tables and per-step details.

Every policy runs the same test days with the same event seeds; each episode is scored with the
optimisation pipeline's ``compute_kpis`` on the episode's dispatch and power flow, plus the EV
KPIs and the reward split by term. ``rl_kpis.csv`` (one row per policy and day) and
``details/<policy>.npz`` (same keys as ``scenarios.run_scenario``, plus the charger powers)
feed ``plots_rl``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.kpis import compute_ev_kpis, compute_kpis
from app.case_studies.kfki_campus.rl.env import KfkiChargingEnv
from app.infra.logging_setup import get_logger
from app.operation.rl.agent import RLAgent
from app.operation.rl.rollout import EpisodeResult, run_episode

log = get_logger(__name__)


@dataclass
class Evaluation:
    """Result of ``evaluate_policies``.

    :ivar pd.DataFrame kpis: One row per (policy, day)
    :ivar dict episodes: ``(policy, day)`` -> dict with ``dispatch``, ``ev_power``,
        ``sessions``, ``vm_pu`` (time x bus) and ``trafo_loading``
    """

    kpis: pd.DataFrame
    episodes: dict = field(default_factory=dict)


def episode_kpis(env: KfkiChargingEnv, policy: str, res: EpisodeResult) -> dict:
    """KPI row of the episode that ``env`` has just run."""
    disp, sim = env.dispatch(), env.sim_result()
    v_min, v_max, v_strict = env.v_limits
    row = {"policy": policy, "day": res.day, "reward": res.reward}
    row.update(compute_kpis(disp, sim, env.dt_h, v_min, v_max, v_strict))
    row.update(compute_ev_kpis(disp, env.ev_power(), env.sessions(), env.dt_h))
    row.update({f"r_{k}": v for k, v in res.terms.items()})
    return row


def evaluate_policies(
    env: KfkiChargingEnv,
    agents: dict[str, RLAgent],
    days: Optional[Sequence[int]] = None,
    seed: int = 0,
    keep_episodes: bool = True,
) -> Evaluation:
    """Run every agent on the same days (default: the env's test days) and score it."""
    days = list(env.test_days if days is None else days)
    rows, episodes = [], {}
    for name, agent in agents.items():
        for i, day in enumerate(days):
            res = run_episode(env, agent, {"day": day}, seed + i, deterministic=True)
            rows.append(episode_kpis(env, name, res))
            if keep_episodes:
                sim = env.sim_result()
                episodes[(name, day)] = dict(
                    dispatch=env.dispatch(),
                    ev_power=env.ev_power(),
                    sessions=env.sessions(),
                    action=env.history("action").copy(),
                    connected=env.history("connected").copy(),
                    vm_pu=sim.vm_pu,
                    trafo_loading=sim.trafo_loading_pct,
                )
        log.info(f"evaluated '{name}' on {len(days)} days")
    return Evaluation(pd.DataFrame(rows), episodes)


def save_details(env: KfkiChargingEnv, path: Path) -> Path:
    """Per-step arrays of the episode just run (keys of ``scenarios.run_scenario``)."""
    disp, sim = env.dispatch(), env.sim_result()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        index=env.window_index.values.astype("datetime64[s]").astype(np.int64),
        vm_max=np.nanmax(sim.vm_pu, axis=1),
        vm_min=np.nanmin(sim.vm_pu, axis=1),
        vm_pu=sim.vm_pu,
        bus_names=np.array(sim.bus_names),
        trafo_loading=sim.trafo_loading_pct,
        trafo_names=np.array(sim.trafo_names),
        demand=disp["demand"].to_numpy(),
        pv=disp["pv"].to_numpy(),
        p_charge=disp["p_charge"].to_numpy(),
        p_discharge=disp["p_discharge"].to_numpy(),
        soc_kwh=disp["soc_kwh"].to_numpy(),
        p_import=disp["p_import"].to_numpy(),
        p_export=disp["p_export"].to_numpy(),
        ev_power=env.ev_power().to_numpy(),
        charger_ids=np.array(list(env.chargers)),
    )
    return path


def full_period_run(
    env: KfkiChargingEnv, agent: RLAgent, name: str, out_dir: Path, seed: int = 0
) -> dict:
    """One continuous run over the whole data period; writes ``details/<name>.npz``."""
    res = run_episode(env, agent, {"full_period": True}, seed)
    save_details(env, Path(out_dir) / "details" / f"{name}.npz")
    return episode_kpis(env, name, res)
