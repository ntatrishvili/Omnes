import pandapower as pp
import pytest

from app.case_studies.kfki_campus.build_model import (
    Scenario,
    allocate,
    build_kfki_model,
)
from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.synthetic import make_synthetic_csv
from app.conversion.pandapower_converter import PandapowerConverter
from app.model.generator.pv import PV
from app.model.storage.battery import Battery

TWO_DAYS = dict(time_start="2025-01-01 00:00", time_end="2025-01-03 00:00")


@pytest.fixture(scope="module")
def data():
    return load_campus_data()


@pytest.fixture(scope="module")
def csv(tmp_path_factory, data):
    return make_synthetic_csv(
        tmp_path_factory.mktemp("kfki") / "ts.csv", days=2, data=data
    )


def test_excel_inputs_match_report(data):
    # five 1 MVA 10/0.4 kV transformers as drawn in Fig. 9 of the report
    assert len(data.transformers) == 5
    assert (data.transformers["sn_kVA"] == 1000).all()
    b = data.battery.iloc[0]
    assert (b.power_kW, b.capacity_kWh) == (250.0, 1450.0)
    pv = data.pv.iloc[0]
    assert pv.base_kWp == 9.0 and pv.n_inverters == 3
    assert len(data.heat_pumps) == 4


def test_allocate_sums_to_total():
    for total in (1, 7, 10, 33):
        assert sum(allocate(total, [0.25] * 4)) == total
    assert allocate(0, [0.5, 0.5]) == [0, 0]


def test_baseline_has_no_added_assets(csv, data):
    m = build_kfki_model(Scenario(0, 1450, 0), csv, data=data, **TWO_DAYS)
    assert len(m.find_all_of_type_in_subentities("PV")) == 1
    assert len(m.find_all_of_type_in_subentities("Battery")) == 1


def test_scenario_scales_assets(csv, data):
    m = build_kfki_model(Scenario(500, 2900, 10), csv, data=data, **TWO_DAYS)
    pvs = [e for e in m.entities.values() if isinstance(e, PV)]
    assert sum(p.peak_power.value for p in pvs) == pytest.approx(9 + 500)
    bat = next(e for e in m.entities.values() if isinstance(e, Battery))
    assert bat.capacity.value == 2900
    assert bat.max_charge_rate.value == pytest.approx(2900 * 250 / 1450)


def _ev_buses(model):
    from app.case_studies.kfki_campus.components import EVCharger

    evs = [e for e in model.entities.values() if isinstance(e, EVCharger)]
    return {e.id: e.bus for e in evs}


def test_chargers_are_nested_round_robin_over_the_sites(csv, data):
    sites = data.ev_sites["bus"].tolist()
    m5 = _ev_buses(build_kfki_model(Scenario(0, 0, 5), csv, data=data, **TWO_DAYS))
    m3 = _ev_buses(build_kfki_model(Scenario(0, 0, 3), csv, data=data, **TWO_DAYS))
    assert list(m5.values()) == [sites[0], sites[1], sites[2], sites[3], sites[0]]
    # the sweep is nested: the first three chargers are the same in both scenarios
    assert all(m5[k] == v for k, v in m3.items()) and len(m3) == 3


def test_each_charger_has_its_own_profile(csv, data):
    m = build_kfki_model(Scenario(0, 0, 4), csv, data=data, **TWO_DAYS)
    cols = {
        e.id: e.p_cons.value(time_set=48, freq="1h")
        for e in m.entities.values()
        if e.__class__.__name__ == "EVCharger"
    }
    assert len(cols) == 4
    sums = {round(float(v.sum()), 3) for v in cols.values()}
    assert len(sums) == 4  # distinct profiles, not one profile scaled


def test_too_many_chargers_is_rejected(csv, data):
    with pytest.raises(ValueError, match="charger profiles"):
        build_kfki_model(Scenario(0, 0, 11), csv, data=data, **TWO_DAYS)


def test_no_battery_when_zero(csv, data):
    m = build_kfki_model(Scenario(0, 0, 0), csv, data=data, **TWO_DAYS)
    assert not m.find_all_of_type_in_subentities("Battery")


def test_pandapower_net_converges(csv, data):
    m = build_kfki_model(Scenario(250, 1450, 10), csv, data=data, **TWO_DAYS)
    net = PandapowerConverter().convert_model(m)
    assert len(net.trafo) == 5 and len(net.bus) == len(data.buses)
    assert len(net.load) == len(data.loads) + len(data.heat_pumps) + 10
    pp.runpp(net)
    assert net.res_bus.vm_pu.between(0.9, 1.1).all()


def test_heat_coupling_adds_converter_tank_and_demand(csv, data):
    from app.case_studies.kfki_campus.components import HeatDemand
    from app.model.storage.hot_water_storage import HotWaterStorage
    from app.model.transducer.transducer import Transducer

    m = build_kfki_model(
        Scenario(0, 1450, 0, heat_coupling=True), csv, data=data, **TWO_DAYS
    )
    ents = list(m.entities.values())
    for cls in (Transducer, HotWaterStorage, HeatDemand):
        assert sum(isinstance(e, cls) for e in ents) == len(data.heat_pumps)
    conv = m.entities["hp_29_conv"]
    assert conv.conversion_efficiency.value == 3.5
    assert conv.output_device == "hp_29_tes"
    # the power flow only sees the electrical side: same net as without heat coupling
    plain = build_kfki_model(Scenario(0, 1450, 0), csv, data=data, **TWO_DAYS)
    net = PandapowerConverter().convert_model(m)
    ref = PandapowerConverter().convert_model(plain)
    assert len(net.load) == len(ref.load) and len(net.sgen) == len(ref.sgen)


def test_scenario_id_marks_heat_coupling():
    assert Scenario(100, 1450, 3).id == "pv100_bat1450_ev3"
    assert Scenario(100, 1450, 3, True, 0.5).id == "pv100_bat1450_ev3_hp0.5"
