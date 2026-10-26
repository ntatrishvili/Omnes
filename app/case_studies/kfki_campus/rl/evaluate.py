"""Evaluation of policies on the KFKI environment: KPI tables and per-step details.

Every policy runs the same test days with the same event seeds; each episode is scored with the
optimisation pipeline's ``compute_kpis`` on the episode's dispatch and power flow, plus the EV
KPIs, the line / transformer overload and the reward split by term (one row per policy and day).
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.kpis import compute_ev_kpis, compute_kpis
from app.case_studies.kfki_campus.rl.env import KfkiChargingEnv
from app.case_studies.kfki_campus.rl.profiles import PROFILES
from app.infra.logging_setup import get_logger
from app.operation.rl.agent import RLAgent
from app.operation.rl.rollout import EpisodeResult, run_episode

log = get_logger(__name__)


@dataclass
class Evaluation:
    """Result of ``evaluate_policies``.

    :ivar pd.DataFrame kpis: One row per (policy, day)
    :ivar dict episodes: ``(policy, day)`` -> dict with ``dispatch``, ``ev_power``,
        ``sessions``, ``vm_pu`` (time x bus), ``trafo_loading`` and ``line_loading``
    """

    kpis: pd.DataFrame
    episodes: dict = field(default_factory=dict)


def line_kpis(env: KfkiChargingEnv, sim) -> dict:
    """Line overloading of the episode: all lines and the feeders that carry chargers.

    ``line_overload_kvah`` is the apparent energy above the thermal rating (the quantity of
    the ``line_overload_kvah`` reward term); ``ev_line_*`` only counts the chargers' feeders;
    ``trafo_overload_kvah`` is measured against the operating limits of
    ``problem.trafo_limits_kva`` (0 without limits).
    """
    load = np.asarray(sim.line_loading_pct, float)
    over = np.maximum(np.nan_to_num(load) - 100.0, 0.0) / 100.0 * env._line_sn
    ev_lines = np.unique(env._ev_line[env._ev_line >= 0])
    # transformers with an operating limit: energy above the limit (0 without limits)
    tl = np.nan_to_num(np.asarray(sim.trafo_loading_pct, float)) * (
        env._trafo_sn / env._trafo_eff_sn
    )
    t_over = np.maximum(tl - 100.0, 0.0) / 100.0 * env._trafo_eff_sn
    t_over = t_over[:, env._trafo_limited]
    return {
        "trafo_overload_kvah": float(t_over.sum() * env.dt_h),
        "line_overload_kvah": float(over.sum() * env.dt_h),
        "ev_line_overload_kvah": float(over[:, ev_lines].sum() * env.dt_h),
        "ev_line_max_loading_pct": (
            float(np.nanmax(load[:, ev_lines])) if load.size else np.nan
        ),
        "line_overload_hours": float((np.nan_to_num(load) > 100.0).any(axis=1).sum() * env.dt_h),
    }


def add_relative_reward(kpis: pd.DataFrame, ref: str = "uncontrolled") -> pd.DataFrame:
    """Add ``reward_vs_uncontrolled``: reward minus that of ``ref`` on the same day.

    The reward of a day is dominated by the part no EV policy can change (PV, campus load);
    the difference to the uncontrolled run is what the policy gained or lost.
    """
    if ref not in set(kpis["policy"]):
        return kpis
    base = kpis[kpis["policy"] == ref].set_index("day")["reward"]
    kpis = kpis.copy()
    kpis["reward_vs_uncontrolled"] = kpis["reward"] - kpis["day"].map(base)
    return kpis


# KPI column that measures each reward metric of a profile
PROFILE_COLUMNS = dict(
    import_kwh="import_kwh",
    export_kwh="export_kwh",
    unmet_kwh="ev_unmet_kwh",
    line_overload_kvah="ev_line_overload_kvah",
    trafo_overload_kvah="trafo_overload_kvah",
)


def add_profile_gains(kpis: pd.DataFrame, ref: str = "uncontrolled") -> pd.DataFrame:
    """Add ``gain_<profile>``: cost saved against ``ref`` on the same day under each profile.

    Cost = sum of weight x KPI over the profile's terms (kWh / kVAh-equivalents per day), so
    every policy can be scored with every profile's weights, whichever weights it was
    trained with. Positive = better than uncontrolled charging.
    """
    if ref not in set(kpis["policy"]):
        return kpis
    kpis = kpis.copy()
    for name, w in PROFILES.items():
        cost = sum(
            w.get(m, 0.0) * kpis[col]
            for m, col in PROFILE_COLUMNS.items()
            if col in kpis
        )
        base = cost[kpis["policy"] == ref].groupby(kpis["day"]).first()
        kpis[f"gain_{name}"] = kpis["day"].map(base) - cost
    return kpis


def episode_kpis(env: KfkiChargingEnv, policy: str, res: EpisodeResult) -> dict:
    """KPI row of the episode that ``env`` has just run."""
    disp, sim = env.dispatch(), env.sim_result()
    v_min, v_max, v_strict = env.v_limits
    row = {"policy": policy, "day": res.day, "reward": res.reward}
    row.update(compute_kpis(disp, sim, env.dt_h, v_min, v_max, v_strict))
    row.update(compute_ev_kpis(disp, env.ev_power(), env.sessions(), env.dt_h))
    row["ev_grid_kwh"] = row["ev_energy_kwh"] * (1.0 - np.nan_to_num(row["ev_pv_share"]))
    row.update(line_kpis(env, sim))
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
                    line_loading=sim.line_loading_pct,
                    line_names=list(env.net.line.name),
                    ev_lines=np.unique(env._ev_line[env._ev_line >= 0]),
                )
        log.info(f"evaluated '{name}' on {len(days)} days")
    kpis = add_profile_gains(add_relative_reward(pd.DataFrame(rows)))
    return Evaluation(kpis, episodes)
