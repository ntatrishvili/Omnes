"""Converter-driven optimisation (EnergySystemOptimizer on top of PulpConverter)."""

import numpy as np
import pandas as pd
import pytest

from app.case_studies.kfki_campus.build_model import Scenario, build_kfki_model
from app.case_studies.kfki_campus.dispatch import BatteryParams, rolling_dispatch
from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.synthetic import make_synthetic_csv
from app.operation.example_optimization import EnergySystemOptimizer

TWO_DAYS = dict(time_start="2025-01-01 00:00", time_end="2025-01-03 00:00")


@pytest.fixture(scope="module")
def data():
    return load_campus_data()


@pytest.fixture(scope="module")
def csv(tmp_path_factory, data):
    return make_synthetic_csv(
        tmp_path_factory.mktemp("opt") / "ts.csv", days=2, data=data
    )


def _optimizer(csv, data, scenario):
    model = build_kfki_model(scenario, csv, data=data, **TWO_DAYS)
    return EnergySystemOptimizer(model), pd.DatetimeIndex(model.time_set.time_points)


def test_relations_of_the_transducer_become_constraints(csv, data):
    opt, _ = _optimizer(csv, data, Scenario(500, 1450, 3, heat_coupling=True))
    # COP relation and rated power of each of the four heat pumps, one constraint per step
    assert len(opt.relations) == 2 * len(data.heat_pumps)
    assert all(len(c) == opt.n for c in opt.relations.values())


def test_heat_pump_follows_cop_and_balances_hold(csv, data):
    opt, idx = _optimizer(csv, data, Scenario(500, 1450, 3, heat_coupling=True))
    sol = opt.solve(0, 48)
    for c in opt.converters:
        cop = c.conversion_efficiency.value
        assert np.allclose(sol.converter_out[c.id], cop * sol.converter_in[c.id], atol=1e-6)
    disp = opt.rolling(idx, 24, 24)
    assert np.allclose(
        disp.heat_demand, disp.heat_supply + disp.tes_discharge - disp.tes_charge, atol=1e-5
    )
    lhs = disp.pv + disp.p_import + disp.p_discharge
    assert np.allclose(lhs, disp.demand + disp.p_charge + disp.p_export, atol=1e-5)
    assert disp.soc_kwh.between(0, 1450 + 1e-6).all()
    assert (disp.p_hp > 0).any()


def test_matches_handwritten_dispatch_without_heat(csv, data):
    opt, idx = _optimizer(csv, data, Scenario(500, 1450, 3))
    new = opt.rolling(idx, 24, 24)
    b = data.battery.iloc[0]
    bat = BatteryParams(
        1450.0,
        250.0,
        250.0,
        float(b.eff_charge),
        float(b.eff_discharge),
        float(b.soc_min),
        float(b.soc_max),
        float(b.soc_init),
    )
    old = rolling_dispatch(idx, new.demand.to_numpy(), new.pv.to_numpy(), bat, 24, 24)
    assert new.p_import.sum() == pytest.approx(old.p_import.sum(), rel=0.01)
    assert new.p_export.sum() == pytest.approx(old.p_export.sum(), rel=0.01, abs=1.0)


def test_battery_does_not_fight_the_grid_exchange(csv, data):
    """No discharge while the campus exports, no charge while it imports (both dispatches)."""
    opt, idx = _optimizer(csv, data, Scenario(8000, 1450, 3))
    new = opt.rolling(idx, 24, 24)
    b = data.battery.iloc[0]
    bat = BatteryParams(
        1450.0, 250.0, 250.0, float(b.eff_charge), float(b.eff_discharge),
        float(b.soc_min), float(b.soc_max), float(b.soc_init),
    )  # fmt: skip
    old = rolling_dispatch(idx, new.demand.to_numpy(), new.pv.to_numpy(), bat, 24, 24)
    for disp in (new, old):
        assert disp.p_export.sum() > 0 and disp.p_discharge.sum() > 0
        assert not ((disp.p_discharge > 1e-3) & (disp.p_export > 1e-3)).any()
        assert not ((disp.p_charge > 1e-3) & (disp.p_import > 1e-3)).any()


def test_dispatch_is_written_back_into_the_model(csv, data):
    """The rolling dispatch is stored in the entities' quantities and readable as a run view."""
    # a large PV plant so that the battery cycles and surplus is exported (non-trivial dispatch)
    opt, idx = _optimizer(csv, data, Scenario(8000, 1450, 3, heat_coupling=True))
    disp = opt.rolling(idx, 24, 24)
    assert disp.p_discharge.sum() > 0 and disp.p_export.sum() > 0
    run = opt.run_view()
    bat = opt.batteries[0]
    assert np.allclose(run[bat.id].p_out.value, disp.p_discharge)
    assert np.allclose(run[bat.id].p_in.value, disp.p_charge)
    assert np.allclose(run[bat.id].e_stor.value, disp.soc_kwh)
    conv = opt.converters[0]
    assert np.allclose(run[conv.id].p_in.value, disp[f"hp_{conv.input_device}"])
    assert np.allclose(run[conv.input_device].p_cons.value, run[conv.id].p_in.value)
    assert np.allclose(run[opt.slacks[0].id].p_out.value, disp.p_import)
    assert np.allclose(run[opt.slacks[0].id].p_in.value, disp.p_export)
    # results live in the run only: the entity's own time series stays empty
    assert bat.quantities["p_out"].empty()
