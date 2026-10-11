import numpy as np
import pandas as pd
import pytest

from app.case_studies.kfki_campus.excel_io import PLOT_MERGE, load_campus_data
from app.case_studies.kfki_campus.plots_network import (
    build_layout,
    mean_kw_by_bus,
    network_topology,
    network_voltage_maps,
)
from app.case_studies.kfki_campus.plots_timeseries import (
    input_carpets,
    input_daily_profiles,
    scenario_week,
    split_loads,
    voltage_duration,
)
from app.case_studies.kfki_campus.synthetic import make_synthetic_csv


@pytest.fixture(scope="module")
def data():
    return load_campus_data()


@pytest.fixture(scope="module")
def csv(tmp_path_factory, data):
    p = make_synthetic_csv(
        tmp_path_factory.mktemp("figs") / "ts.csv", "2025-06-16", 14, data=data
    )
    return pd.read_csv(p, index_col=0, parse_dates=True)


def test_layout_places_every_feeder_once_and_in_order(data):
    lay = build_layout(data)
    assert len(lay.terminals) == len(set(lay.terminals))
    # every line target that hangs off an LV busbar is a terminal, except feeders that are
    # drawn as part of another building (the tenant's feeder, merged into building 21)
    lv = set(data.transformers["to_bus"])
    targets = set(data.lines.loc[data.lines["from_bus"].isin(lv), "to_bus"])
    hidden = set(data.loads.loc[data.loads["id"].isin(PLOT_MERGE), "bus"])
    assert hidden and set(lay.terminals) == targets - hidden
    xs = [lay.x[t] for t in lay.terminals]
    assert xs == sorted(xs) and len(set(xs)) == len(xs)  # distinct, left to right
    for lv_bus, (x0, x1) in lay.bars.items():  # each busbar spans its own feeders
        kids = [t for t in lay.terminals if lay.group_of[t] == lv_bus]
        assert x0 < min(lay.x[k] for k in kids) and x1 > max(lay.x[k] for k in kids)


def test_plot_labels_are_english(data):
    from app.case_studies.kfki_campus.excel_io import display_names
    from app.case_studies.kfki_campus.plots_network import pretty_bus

    load_names, bus_names = display_names(data)
    assert load_names["villanyszerelo_muhely"] == "Electrician's workshop"
    assert load_names["b17"] == "Bldg 17" and load_names["reaktor_1"] == "Reactor 1"
    assert bus_names["4T_nas"] == "NaS battery"
    assert pretty_bus("MV_10kV", bus_names) == "10 kV bus"
    assert pretty_bus("LV_4T", bus_names) == "LV bus 4T"
    # the tenant is shown as part of building 21, and no Hungarian letters or the
    # abbreviation 'ep.' are left in the plot labels
    assert load_names["b21A"] == "Bldg 21" and load_names["kovacs_laszlo"] == "Bldg 21"
    text = " ".join(list(load_names.values()) + list(bus_names.values()))
    assert "Kov" not in text
    assert not any(ch in text for ch in "áéíóöőúüűÁÉÍÓÖŐÚÜŰ")
    assert " ép" not in text and " ep." not in text


def test_merged_load_is_added_to_its_target_bus(data, csv):
    m = mean_kw_by_bus(data, csv)
    tenant = data.loads.set_index("id").loc["kovacs_laszlo"]
    target = data.loads.set_index("id").loc["b21A"]
    assert tenant.bus not in m  # not drawn on its own
    expected = csv[target.csv_column].mean() + csv[tenant.csv_column].mean()
    for hp in data.heat_pumps[data.heat_pumps["bus"] == target.bus].itertuples():
        expected += csv[hp.csv_column].mean()  # the heat pump on the same bus
    assert m[target.bus] == pytest.approx(expected)


def test_mean_load_by_bus_matches_csv(data, csv):
    m = mean_kw_by_bus(data, csv)
    r = data.loads.iloc[0]
    assert m[r.bus] == pytest.approx(csv[r.csv_column].mean())


def test_split_loads_adds_up(data, csv):
    mapped = set(data.loads["id"].iloc[:3])
    loads = split_loads(csv, data, mapped)
    assert np.allclose(loads["total"], loads["metered"] + loads["unmetered"])
    assert (loads["unmetered"] >= -1e-9).all()


def test_topology_and_input_figures_are_written(data, csv, tmp_path):
    mapped = set(data.loads["id"].iloc[:3])
    loads = split_loads(csv, data, mapped)
    for f in (
        network_topology(data, csv, mapped, tmp_path / "topo.png"),
        input_daily_profiles(csv, loads, tmp_path / "prof.png"),
        input_carpets(csv, loads, tmp_path / "carpet.png"),
    ):
        assert f.exists() and f.stat().st_size > 5000


def _fake_run(path, n=24 * 14, with_battery=True):
    t = pd.date_range("2025-06-16", periods=n, freq="1h")
    h = t.hour.to_numpy()
    pv = np.clip(np.sin(np.pi * (h - 6) / 12), 0, None) * 1000
    np.savez(
        path,
        index=t.values.astype("datetime64[s]").astype(np.int64),
        vm_max=1.0 + 0.00003 * pv,
        vm_min=np.full(n, 0.97),
        vm_pu=np.full((n, 3), 1.0),
        bus_names=np.array(["MV_10kV", "LV_T", "T_b17"]),
        trafo_loading=np.full((n, 5), 30.0),
        trafo_names=np.array(["TR_T", "TR_3T", "TR_4T", "TR_6T", "TR_10T"]),
        demand=np.full(n, 800.0),
        pv=pv,
        p_charge=np.where(pv > 800, 300.0, 0.0) if with_battery else np.zeros(n),
        p_discharge=np.zeros(n),
        soc_kwh=(
            np.cumsum(np.where(pv > 800, 300.0, 0.0)) if with_battery else np.zeros(n)
        ),
        p_import=np.clip(800 - pv, 0, None),
        p_export=np.clip(pv - 800, 0, None),
    )


def test_scenario_week_and_voltage_maps(data, csv, tmp_path):
    det = tmp_path / "details"
    det.mkdir()
    _fake_run(det / "pv500_bat1450_ev0.npz")
    out = scenario_week(det, "pv500_bat1450_ev0", "2025-06-17", tmp_path / "w.png")
    assert out.exists()
    assert scenario_week(det, "missing", "2025-06-17", tmp_path / "x.png") is None
    maps = network_voltage_maps(
        data, csv, det, [("pv500_bat1450_ev0", "max", "test")], tmp_path / "v.png"
    )
    assert (
        maps is None or maps.exists()
    )  # fake run has 3 buses only: drawn without error


def test_voltage_duration_uses_available_runs(tmp_path):
    det = tmp_path / "details"
    det.mkdir()
    _fake_run(det / "pv500_bat1450_ev0.npz")
    _fake_run(det / "pv1000_bat1450_ev0.npz")
    kpis = pd.DataFrame(
        {
            "scenario": ["pv500_bat1450_ev0", "pv1000_bat1450_ev0"],
            "added_pv_kwp": [500.0, 1000.0],
            "battery_kwh": [1450.0, 1450.0],
            "n_chargers": [0, 0],
        }
    )
    assert voltage_duration(kpis, det, tmp_path / "vd.png").exists()
    assert voltage_duration(kpis.iloc[0:0], det, tmp_path / "none.png") is None
