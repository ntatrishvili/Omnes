"""Rolling-horizon PuLP dispatch of the campus battery and (optionally) the heat pumps.

The campus is treated as a copper plate for the energy balance (the network is
handled afterwards by the Pandapower simulation); the only network constraint
is the apparent-power limit of the transformer the battery is connected to.

Objective per window: minimise grid import + export (i.e. maximise
self-consumption and self-sufficiency) plus a tiny throughput penalty that
prevents simultaneous charging and discharging.

Electricity-heat coupling: each heat pump converts electric power ``hp_e`` into heat
(``cop * hp_e``) that charges a hot-water tank; the tank and the heat pump together serve
the building's heat demand (heat balance). The heat pump electricity is part of the
campus demand, so the optimiser shifts it into hours of PV surplus and cheap grid use.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pulp

THROUGHPUT_PENALTY = 1e-3  # kW-equivalent cost per kW of battery throughput
TRAFO_PENALTY = 10.0  # cost per kW above the transformer limit (>> import/export cost)
UNMET_HEAT_PENALTY = 100.0  # cost per kW of heat demand left unserved (>> import cost)
TES_REFILL_PENALTY = 1.0  # cost per kWh a tank ends below its initial level (> 1/COP)


@dataclass(frozen=True)
class BatteryParams:
    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    eff_charge: float = 0.9
    eff_discharge: float = 0.9
    soc_min: float = 0.0  # fractions of capacity
    soc_max: float = 1.0
    soc_init: float = 0.5


@dataclass(frozen=True)
class HeatParams:
    """One heat pump + hot-water tank (thermal kW / kWh, electric kW)."""

    id: str
    p_max_kw: float  # electric input limit of the heat pump
    cop: float  # constant coefficient of performance
    tes_capacity_kwh: float
    tes_max_rate_kw: float  # tank charge / discharge limit
    tes_loss_per_h: float = 0.005  # standing loss, share of the stored heat per hour
    tes_soc_init: float = 0.5  # fraction of the capacity
    in_battery_trafo: bool = False  # electric bus on the battery's transformer


@dataclass
class WindowResult:
    ch: np.ndarray
    dis: np.ndarray
    soc: np.ndarray
    hp_e: np.ndarray  # (n, K) heat pump electric power
    tes_soc: np.ndarray  # (n, K) tank state of charge (kWh)
    tes_ch: np.ndarray  # (n, K) heat into the tank
    tes_dis: np.ndarray  # (n, K) heat out of the tank
    unmet: np.ndarray  # (n, K) heat demand not served


def _solve_window(
    demand: np.ndarray,
    pv: np.ndarray,
    bat: BatteryParams,
    soc0_kwh: float,
    dt_h: float,
    trafo_base_kw: Optional[np.ndarray],
    trafo_limit_kw: Optional[float],
    solver,
    heat: tuple = (),
    heat_demand: Optional[np.ndarray] = None,
    tes0_kwh: Optional[np.ndarray] = None,
) -> WindowResult:
    """Solve one window.

    The plain LP is solved first. If it burns PV surplus through the battery's
    round-trip losses (discharging while the campus exports, charging while it
    imports, or both in one step), the MILP is solved instead: the battery may only
    discharge in steps without export and only charge in steps without import.
    Most windows have no surplus, so this is much faster.
    """
    args = (
        demand,
        pv,
        bat,
        soc0_kwh,
        dt_h,
        trafo_base_kw,
        trafo_limit_kw,
        solver,
        heat,
        heat_demand,
        tes0_kwh,
    )
    res = _solve_window_once(*args, exclusive=False)
    # + = import, - = export, with the battery and the heat pumps included
    balance = demand + res.hp_e.sum(axis=1) + res.ch - pv - res.dis
    if (
        np.minimum(res.ch, res.dis).max() > 1e-6
        or ((res.ch > 1e-3) & (balance > 1e-3)).any()
        or ((res.dis > 1e-3) & (balance < -1e-3)).any()
    ):
        res = _solve_window_once(*args, exclusive=True)
    return res


def _solve_window_once(
    demand: np.ndarray,
    pv: np.ndarray,
    bat: BatteryParams,
    soc0_kwh: float,
    dt_h: float,
    trafo_base_kw: Optional[np.ndarray],
    trafo_limit_kw: Optional[float],
    solver,
    heat: tuple,
    heat_demand: Optional[np.ndarray],
    tes0_kwh: Optional[np.ndarray],
    exclusive: bool,
) -> WindowResult:
    n = len(demand)
    K = len(heat)
    prob = pulp.LpProblem("kfki_dispatch", pulp.LpMinimize)
    ch = [pulp.LpVariable(f"ch_{t}", 0, bat.max_charge_kw) for t in range(n)]
    dis = [pulp.LpVariable(f"dis_{t}", 0, bat.max_discharge_kw) for t in range(n)]
    imp = [pulp.LpVariable(f"imp_{t}", 0) for t in range(n)]
    exp = [pulp.LpVariable(f"exp_{t}", 0) for t in range(n)]
    lo, hi = bat.soc_min * bat.capacity_kwh, bat.soc_max * bat.capacity_kwh
    soc = [pulp.LpVariable(f"soc_{t}", lo, hi) for t in range(n)]  # soc after step t

    # heat pumps and tanks, indexed [k][t]
    hp_e = [
        [pulp.LpVariable(f"hpe_{k}_{t}", 0, h.p_max_kw) for t in range(n)]
        for k, h in enumerate(heat)
    ]
    tes_ch = [
        [pulp.LpVariable(f"tch_{k}_{t}", 0, h.tes_max_rate_kw) for t in range(n)]
        for k, h in enumerate(heat)
    ]
    tes_dis = [
        [pulp.LpVariable(f"tdi_{k}_{t}", 0, h.tes_max_rate_kw) for t in range(n)]
        for k, h in enumerate(heat)
    ]
    tes_soc = [
        [pulp.LpVariable(f"tso_{k}_{t}", 0, h.tes_capacity_kwh) for t in range(n)]
        for k, h in enumerate(heat)
    ]
    unmet = [
        [pulp.LpVariable(f"unm_{k}_{t}", 0) for t in range(n)] for k in range(K)
    ]
    hp_total = [pulp.lpSum(hp_e[k][t] for k in range(K)) for t in range(n)]

    objective = pulp.lpSum(
        (imp[t] + exp[t]) * dt_h + THROUGHPUT_PENALTY * (ch[t] + dis[t]) * dt_h
        for t in range(n)
    ) + UNMET_HEAT_PENALTY * pulp.lpSum(u for row in unmet for u in row) * dt_h
    over = []  # transformer-limit slacks (soft constraint, see below)
    # The battery discharges only while the campus imports and charges only while it
    # exports. Without this the LP burns PV surplus through round-trip losses (discharging
    # into an exporting campus and recharging later, or charging and discharging at once),
    # which a min(import + export) objective rewards but a real operation would not do.
    if exclusive:
        grid_import = [  # 1 = import step (no export), 0 = export step (no import)
            pulp.LpVariable(f"grid_import_{t}", cat=pulp.LpBinary) for t in range(n)
        ]
        big = (
            3.0 * (float(np.max(demand)) + float(np.max(pv)))
            + sum(h.p_max_kw for h in heat)
            + bat.max_charge_kw
            + bat.max_discharge_kw
            + 1e3
        )
        for t in range(n):
            prob += imp[t] <= big * grid_import[t]
            prob += exp[t] <= big * (1 - grid_import[t])
            prob += ch[t] <= bat.max_charge_kw * (1 - grid_import[t])
            prob += dis[t] <= bat.max_discharge_kw * grid_import[t]
    for t in range(n):
        prev = soc0_kwh if t == 0 else soc[t - 1]
        prob += (
            soc[t]
            == prev + bat.eff_charge * ch[t] * dt_h - dis[t] * dt_h / bat.eff_discharge
        )
        prob += pv[t] + imp[t] + dis[t] == demand[t] + hp_total[t] + ch[t] + exp[t]
        if trafo_base_kw is not None and trafo_limit_kw is not None:
            hp_on_trafo = pulp.lpSum(
                hp_e[k][t] for k, h in enumerate(heat) if h.in_battery_trafo
            )
            net = (
                trafo_base_kw[t] + hp_on_trafo + ch[t] - dis[t]
            )  # + = demand drawn through the trafo
            # Soft limit: with a lot of added PV on this transformer the base flow alone
            # can exceed it (battery cannot fix that); the overload is then reported by
            # the power-flow simulation instead of making the dispatch infeasible.
            up = pulp.LpVariable(f"tu_{t}", 0)
            dn = pulp.LpVariable(f"td_{t}", 0)
            prob += net <= trafo_limit_kw + up
            prob += net >= -trafo_limit_kw - dn
            over += [up, dn]
    refill = []  # tank shortfall at the window end (see below)
    for k, h in enumerate(heat):
        keep = 1.0 - h.tes_loss_per_h * dt_h
        for t in range(n):
            prev = tes0_kwh[k] if t == 0 else tes_soc[k][t - 1]
            prob += tes_soc[k][t] == keep * prev + (tes_ch[k][t] - tes_dis[k][t]) * dt_h
            # heat balance of the building: heat pump + tank discharge = demand + tank charge
            prob += (
                h.cop * hp_e[k][t] + tes_dis[k][t] + unmet[k][t]
                == heat_demand[t, k] + tes_ch[k][t]
            )
        # no heat mining across the window edge: end at least at the initial level. Soft, so
        # the tank can start a window below it (it is then refilled where cheapest).
        short = pulp.LpVariable(f"tshort_{k}", 0)
        prob += tes_soc[k][n - 1] + short >= h.tes_soc_init * h.tes_capacity_kwh
        refill.append(short)
    prob.setObjective(
        objective
        + TRAFO_PENALTY * pulp.lpSum(over) * dt_h
        + TES_REFILL_PENALTY * pulp.lpSum(refill)
    )
    # no energy mining across the window edge
    prob += soc[n - 1] >= min(soc0_kwh, bat.soc_init * bat.capacity_kwh)

    status = prob.solve(solver)
    if exclusive and pulp.LpStatus[status] == "Infeasible":
        # CBC's pre-processing sometimes declares the (feasible) grid-direction MILP
        # infeasible; the same problem solves with pre-processing off
        status = prob.solve(
            pulp.PULP_CBC_CMD(
                msg=False,
                timeLimit=getattr(solver, "timeLimit", None) or 5,
                gapRel=getattr(solver, "gapRel", None),
                options=["preprocess off"],
            )
        )
    # With a time limit CBC may stop on a feasible (near-optimal) integer solution.
    have_solution = all(v.value() is not None for v in ch + dis + soc)
    if pulp.LpStatus[status] != "Optimal" and not (
        pulp.LpStatus[status] == "Not Solved" and have_solution
    ):
        raise RuntimeError(
            f"dispatch window infeasible/unsolved: {pulp.LpStatus[status]}"
        )
    val = lambda vs: np.array([v.value() for v in vs], dtype=float)  # noqa: E731
    grid = lambda vv: (  # noqa: E731  (K lists of n variables -> n x K array)
        np.array([val(row) for row in vv]).T if K else np.zeros((n, 0))
    )
    return WindowResult(
        val(ch),
        val(dis),
        val(soc),
        grid(hp_e),
        grid(tes_soc),
        grid(tes_ch),
        grid(tes_dis),
        grid(unmet),
    )


def rolling_dispatch(
    index: pd.DatetimeIndex,
    demand_kw: np.ndarray,
    pv_kw: np.ndarray,
    battery: Optional[BatteryParams],
    horizon_steps: int,
    lookahead_steps: int,
    trafo_base_kw: Optional[np.ndarray] = None,
    trafo_limit_kw: Optional[float] = None,
    solver=None,
    heat: tuple = (),
    heat_demand_kw: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Dispatch the battery (and heat pumps) over the whole index in committed blocks.

    Parameters are time series in kW on a regular grid. ``trafo_base_kw`` is the
    net demand (load - PV) already flowing through the battery's transformer
    without the battery (and without the heat pumps). ``heat`` is a tuple of
    ``HeatParams`` and ``heat_demand_kw`` the thermal demand, shape (T, len(heat)).

    Returns a DataFrame (kW, kWh) with columns demand (incl. heat pump electricity),
    pv, p_charge, p_discharge, soc_kwh, p_import, p_export and, with heat pumps,
    ``hp_<id>`` (electric kW per heat pump) plus heat_demand, heat_supply (heat pump
    output), tes_soc_kwh, tes_charge, tes_discharge and heat_unmet (all campus totals).
    """
    T = len(index)
    K = len(heat)
    dt_h = pd.Timedelta(index.freq or (index[1] - index[0])) / pd.Timedelta("1h")
    p_ch, p_dis, soc_out = np.zeros(T), np.zeros(T), np.zeros(T)
    hp_e, tes_soc_out = np.zeros((T, K)), np.zeros((T, K))
    tes_ch, tes_dis, unmet = np.zeros((T, K)), np.zeros((T, K)), np.zeros((T, K))
    has_battery = battery is not None and battery.capacity_kwh > 0
    if K and heat_demand_kw is None:
        raise ValueError("heat pumps given without heat_demand_kw")
    if has_battery or K:
        battery = battery or BatteryParams(0.0, 0.0, 0.0)
        # time/gap limits: a near-optimal feasible dispatch is enough, and the exclusive
        # charge/discharge MILP can otherwise run for minutes on degenerate windows
        solver = solver or pulp.PULP_CBC_CMD(msg=False, timeLimit=5, gapRel=0.02)
        soc = battery.soc_init * battery.capacity_kwh
        tes = np.array([h.tes_soc_init * h.tes_capacity_kwh for h in heat])
        for s in range(0, T, horizon_steps):
            e = min(T, s + horizon_steps + lookahead_steps)
            c = min(horizon_steps, T - s)
            r = _solve_window(
                demand_kw[s:e],
                pv_kw[s:e],
                battery,
                soc,
                dt_h,
                None if trafo_base_kw is None else trafo_base_kw[s:e],
                trafo_limit_kw,
                solver,
                heat,
                None if not K else heat_demand_kw[s:e],
                tes,
            )
            sl = slice(s, s + c)
            p_ch[sl], p_dis[sl], soc_out[sl] = r.ch[:c], r.dis[:c], r.soc[:c]
            hp_e[sl], tes_soc_out[sl] = r.hp_e[:c], r.tes_soc[:c]
            tes_ch[sl], tes_dis[sl], unmet[sl] = r.tes_ch[:c], r.tes_dis[:c], r.unmet[:c]
            soc = r.soc[c - 1]
            if K:
                tes = r.tes_soc[c - 1]
    hp_total = hp_e.sum(axis=1)
    total_demand = demand_kw + hp_total
    balance = total_demand + p_ch - pv_kw - p_dis  # >0 import, <0 export
    out = pd.DataFrame(
        {
            "demand": total_demand,
            "pv": pv_kw,
            "p_charge": p_ch,
            "p_discharge": p_dis,
            "soc_kwh": soc_out,
            "p_import": np.clip(balance, 0, None),
            "p_export": np.clip(-balance, 0, None),
        },
        index=index,
    )
    if K:
        for k, h in enumerate(heat):
            out[f"hp_{h.id}"] = hp_e[:, k]
        cop = np.array([h.cop for h in heat])
        out["p_hp"] = hp_total
        out["heat_demand"] = heat_demand_kw.sum(axis=1)
        out["heat_supply"] = (hp_e * cop).sum(axis=1)
        out["tes_soc_kwh"] = tes_soc_out.sum(axis=1)
        out["tes_charge"] = tes_ch.sum(axis=1)
        out["tes_discharge"] = tes_dis.sum(axis=1)
        out["heat_unmet"] = unmet.sum(axis=1)
        out.attrs["tes_kwh"] = float(sum(h.tes_capacity_kwh for h in heat))
    return out
