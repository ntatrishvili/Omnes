import numpy as np
import pandas as pd
import pytest

from app.case_studies.kfki_campus.dispatch import BatteryParams, rolling_dispatch
from app.case_studies.kfki_campus.kpis import compute_kpis
from app.case_studies.kfki_campus.simulate import SimResult

IDX = pd.date_range("2025-06-01", periods=48, freq="1h")
HOUR = IDX.hour.to_numpy()
PV = np.clip(np.sin(np.pi * (HOUR - 6) / 12), 0, None) * 1000.0  # 1 MW noon peak
DEMAND = np.full(48, 300.0)
BAT = BatteryParams(2000.0, 500.0, 500.0, 0.9, 0.9, 0.0, 1.0, 0.5)


def run(battery, **kw):
    return rolling_dispatch(IDX, DEMAND, PV, battery, 24, 24, **kw)


def test_energy_balance_holds_every_step():
    d = run(BAT)
    lhs = d.pv + d.p_import + d.p_discharge
    rhs = d.demand + d.p_charge + d.p_export
    assert np.allclose(lhs, rhs, atol=1e-6)


def test_soc_within_bounds_and_no_simultaneous_charge_discharge():
    d = run(BAT)
    assert d.soc_kwh.between(0, 2000 + 1e-6).all()
    assert (np.minimum(d.p_charge, d.p_discharge) < 1e-6).all()


def test_battery_reduces_export_and_import():
    without, with_b = run(None), run(BAT)
    assert with_b.p_export.sum() < without.p_export.sum()
    assert with_b.p_import.sum() < without.p_import.sum()
    assert without.p_charge.sum() == 0


def test_soc_trajectory_matches_efficiencies():
    d = run(BAT)
    soc = 0.5 * 2000
    for t in range(24):  # first committed block
        soc += 0.9 * d.p_charge.iloc[t] - d.p_discharge.iloc[t] / 0.9
        assert d.soc_kwh.iloc[t] == pytest.approx(soc, abs=1e-4)


def test_transformer_limit_caps_net_flow():
    base = np.full(48, 900.0)
    d = run(BAT, trafo_base_kw=base, trafo_limit_kw=1000.0)
    assert (base + d.p_charge - d.p_discharge).max() <= 1000.0 + 1e-6


def test_transformer_limit_is_soft_when_base_flow_alone_exceeds_it():
    base = np.full(48, -1600.0)  # 1.6 MW of PV export through a 1 MW transformer
    d = run(BAT, trafo_base_kw=base, trafo_limit_kw=1000.0)  # must not raise
    # the battery still helps: it charges as much as it can to reduce the overload
    assert d.p_charge.max() > 0


def _sim(vm):
    n = len(vm)
    return SimResult(
        index=IDX[:n],
        bus_names=["a", "b"],
        vm_pu=np.asarray(vm, dtype=float),
        trafo_names=["t"],
        trafo_loading_pct=np.full((n, 1), 50.0),
        line_loading_pct=np.full((n, 1), 30.0),
        losses_kw=np.full(n, 10.0),
        ext_grid_p_kw=np.zeros(n),
        converged=np.ones(n, dtype=bool),
    )


def test_kpi_formulas_hand_computed():
    disp = pd.DataFrame(
        {
            "demand": [100.0, 100.0],
            "pv": [50.0, 150.0],
            "p_charge": 0.0,
            "p_discharge": 0.0,
            "soc_kwh": 0.0,
            "p_import": [50.0, 0.0],
            "p_export": [0.0, 50.0],
        },
        index=IDX[:2],
    )
    disp.attrs["battery_kwh"] = 0.0
    k = compute_kpis(disp, _sim([[1.0, 1.0], [1.12, 0.95]]), dt_h=1.0)
    assert k["SC"] == pytest.approx(1 - 50 / 200)  # 25 % of PV exported
    assert k["SS"] == pytest.approx(1 - 50 / 200)
    assert k["n_over"] == 1 and k["n_under"] == 0
    assert k["viol_frac"] == pytest.approx(1 / 4)
    assert k["viol_step_frac"] == pytest.approx(1 / 2)
    assert k["worst_bus"] == "a"
    assert k["losses_kwh"] == pytest.approx(20.0)


def test_kpi_sc_is_nan_without_pv():
    disp = pd.DataFrame(
        {
            "demand": [10.0],
            "pv": [0.0],
            "p_charge": 0.0,
            "p_discharge": 0.0,
            "soc_kwh": 0.0,
            "p_import": [10.0],
            "p_export": [0.0],
        },
        index=IDX[:1],
    )
    disp.attrs["battery_kwh"] = 0.0
    k = compute_kpis(disp, _sim([[1.0, 1.0]]), dt_h=1.0)
    assert np.isnan(k["SC"]) and k["SS"] == 0.0


# ---------------------------------------------------------------- electricity-heat coupling
from app.case_studies.kfki_campus.dispatch import HeatParams  # noqa: E402

HEAT = (
    HeatParams("hp1", p_max_kw=100.0, cop=3.0, tes_capacity_kwh=600.0, tes_max_rate_kw=600.0),
)
HEAT_DEMAND = np.full((48, 1), 90.0)  # kW thermal, constant


def run_heat(heat=HEAT, battery=None):
    return rolling_dispatch(
        IDX, DEMAND, PV, battery, 24, 24, heat=heat, heat_demand_kw=HEAT_DEMAND
    )


def test_heat_balance_and_electric_balance_hold():
    d = run_heat()
    assert (d.heat_unmet < 1e-6).all()
    # heat pump output + tank discharge = demand + tank charge
    heat_in = d.heat_supply + d.tes_discharge
    assert np.allclose(heat_in, d.heat_demand + d.tes_charge, atol=1e-6)
    # heat pump electricity is part of the campus demand
    assert np.allclose(d.demand, DEMAND + d.hp_hp1, atol=1e-6)
    assert np.allclose(d.pv + d.p_import, d.demand + d.p_export, atol=1e-6)
    assert d.hp_hp1.max() <= 100.0 + 1e-6


def test_tank_shifts_heat_pump_into_pv_hours():
    no_tank = run_heat(
        (HeatParams("hp1", 100.0, 3.0, 0.0, 0.0),)
    )
    tank = run_heat()
    # without a tank the pump must follow the (flat) heat demand; with one it follows the PV
    assert no_tank.hp_hp1.std() < 1e-6
    assert tank.p_import.sum() <= no_tank.p_import.sum() + 1e-6
    noon = tank.index.hour.isin(range(10, 15))
    assert tank.hp_hp1[noon].mean() > tank.hp_hp1[~noon].mean()
