"""Scenario KPIs: self-consumption, self-sufficiency, voltage violations.

Definitions (all energy based, campus level, over the simulated period):
  SC  = 1 - E_export / E_pv           share of PV energy used on campus
                                      (directly or via the battery); NaN if no PV
  SS  = 1 - E_import / E_demand       share of campus demand covered locally
  E_demand excludes battery charging; battery losses are therefore covered by
  import/PV and show up as a lower SS.

Voltage violations are counted over all (bus, step) pairs of converged steps:
  viol_frac       fraction outside [v_min, v_max]
  viol_strict_frac fraction above v_max_strict (stricter reporting band)
"""

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.simulate import SimResult


def compute_ev_kpis(
    dispatch: pd.DataFrame, ev_power: pd.DataFrame, sessions: pd.DataFrame, dt_h: float
) -> dict:
    """EV-charging KPIs of a controlled run (kept apart so ``compute_kpis`` stays unchanged).

    ``ev_power`` is the power per charger (kW), ``sessions`` the table of finished sessions
    (``requested_kwh``, ``delivered_kwh``). ``ev_pv_share`` is the part of the EV energy that
    coincides with a PV surplus over the rest of the campus demand.
    """
    ev = ev_power.sum(axis=1).to_numpy()
    base = dispatch["demand"].to_numpy() - ev
    surplus = np.clip(dispatch["pv"].to_numpy() - base, 0.0, None)
    e_ev = float(ev.sum() * dt_h)
    req = float(sessions["requested_kwh"].sum()) if len(sessions) else 0.0
    dlv = float(sessions["delivered_kwh"].sum()) if len(sessions) else 0.0
    return {
        "ev_energy_kwh": e_ev,
        "ev_sessions": int(len(sessions)),
        "ev_requested_kwh": req,
        "ev_unmet_kwh": req - dlv,
        "ev_satisfaction": dlv / req if req > 0 else np.nan,
        "ev_pv_share": (
            float(np.minimum(ev, surplus).sum() * dt_h / e_ev) if e_ev > 0 else np.nan
        ),
        "ev_peak_kw": float(ev.max()) if len(ev) else 0.0,
    }


def compute_kpis(
    dispatch: pd.DataFrame,
    sim: SimResult,
    dt_h: float,
    v_min: float = 0.9,
    v_max: float = 1.1,
    v_max_strict: float = 1.05,
) -> dict:
    e = lambda col: float(dispatch[col].sum() * dt_h)  # noqa: E731  (kWh)
    e_pv, e_dem = e("pv"), e("demand")
    e_imp, e_exp = e("p_import"), e("p_export")
    ok = sim.converged
    vm = sim.vm_pu[ok]
    n_obs = vm.size
    over = (vm > v_max).sum()
    under = (vm < v_min).sum()
    strict = (vm > v_max_strict).sum()
    viol_any_step = ((vm > v_max) | (vm < v_min)).any(axis=1).sum() if n_obs else 0
    worst_bus = sim.bus_names[int(np.nanargmax(np.nanmax(vm, axis=0)))] if n_obs else ""
    cap_cycles = (
        e("p_discharge") / dispatch.attrs["battery_kwh"]
        if dispatch.attrs.get("battery_kwh", 0) > 0
        else 0.0
    )
    heat = {}
    if "heat_demand" in dispatch:  # electricity-heat coupling (heat pumps + tanks)
        e_hp, e_supply = e("p_hp"), e("heat_supply")
        heat = {
            "hp_kwh": e_hp,
            "heat_demand_kwh": e("heat_demand"),
            "heat_supply_kwh": e_supply,
            # heat demand not covered by the heat pumps and the tanks (0 by construction)
            "heat_unmet_kwh": e("heat_demand")
            - e("heat_supply")
            - e("tes_discharge")
            + e("tes_charge"),
            "hp_cop_seasonal": e_supply / e_hp if e_hp > 0 else np.nan,
            "tes_cycles": (
                e("tes_discharge") / dispatch.attrs["tes_kwh"]
                if dispatch.attrs["tes_kwh"] > 0
                else 0.0
            ),
        }
    return {
        "pv_kwh": e_pv,
        "demand_kwh": e_dem,
        "import_kwh": e_imp,
        "export_kwh": e_exp,
        "SC": 1.0 - e_exp / e_pv if e_pv > 0 else np.nan,
        "SS": 1.0 - e_imp / e_dem if e_dem > 0 else np.nan,
        "n_steps": len(ok),
        "n_nonconverged": int((~ok).sum()),
        "v_max_pu": float(np.nanmax(vm)) if n_obs else np.nan,
        "v_min_pu": float(np.nanmin(vm)) if n_obs else np.nan,
        "n_over": int(over),
        "n_under": int(under),
        "viol_frac": float((over + under) / n_obs) if n_obs else np.nan,
        "viol_strict_frac": float(strict / n_obs) if n_obs else np.nan,
        "viol_step_frac": float(viol_any_step / ok.sum()) if ok.any() else np.nan,
        "worst_bus": worst_bus,
        "trafo_max_loading_pct": (
            float(np.nanmax(sim.trafo_loading_pct)) if n_obs else np.nan
        ),
        "line_max_loading_pct": (
            float(np.nanmax(sim.line_loading_pct)) if n_obs else np.nan
        ),
        "losses_kwh": float(np.nansum(sim.losses_kw) * dt_h),
        "battery_cycles": cap_cycles,
        **heat,
    }
