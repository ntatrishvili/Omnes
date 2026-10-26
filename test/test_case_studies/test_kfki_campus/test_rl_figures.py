"""Hungarian figures of the RL case study and its GMM sampling stage, on small synthetic tables."""

import numpy as np
import pandas as pd
import pytest

from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.rl.gmm_events import WORKDAY_FILE
from app.case_studies.kfki_campus.rl import figures_hu as F
from app.case_studies.kfki_campus.rl.run import SENS_AXES, gmm_samples

POLICIES = ["uncontrolled", "pv_following", "line_following", "grid_guard", "oracle"] + [
    f"ppo_{p}_s{s}" for p in ("pv_max", "grid_safe", "balanced") for s in (0, 1)
]
DAYS = [3, 7, 9]


def _kpis(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pol in POLICIES:
        for d in DAYS:
            r = {"policy": pol, "day": d}
            better = {"uncontrolled": 0.0, "oracle": 1.0}.get(pol, 0.5)
            r["ev_pv_share"] = 0.3 + 0.1 * better + 0.01 * rng.random()
            r["ev_line_overload_kvah"] = 40 - 8 * better + rng.random()
            r["trafo_overload_kvah"] = 110 - 100 * better + rng.random()
            r["ev_unmet_kwh"] = 0.0
            r["import_kwh"] = 15000.0
            r["ev_sessions"] = 12
            for prof in ("pv_max", "grid_safe", "balanced", "comfort_first"):
                r[f"gain_{prof}"] = 100.0 * better + rng.random()
            rows.append(r)
    return pd.DataFrame(rows)


def _episodes():
    eps = {}
    idx = pd.date_range("2025-06-02", periods=24, freq="h")
    for pol in POLICIES:
        for d in DAYS:
            eps[(pol, d)] = dict(
                dispatch=pd.DataFrame(
                    dict(
                        pv=np.sin(np.linspace(0, np.pi, 24)) * 1500,
                        demand=np.full(24, 1000.0),
                        p_charge=np.zeros(24),
                        p_discharge=np.zeros(24),
                    ),
                    index=idx,
                ),
                ev_power=pd.DataFrame(np.random.rand(24, 3) * 22, index=idx),
                line_loading=np.random.rand(24, 6) * 120,
                trafo_loading=np.random.rand(24, 5) * 40,
                ev_lines=np.array([1, 2, 3]),
            )
    return eps


def test_labels_are_hungarian_and_seed_free():
    assert F.label("ppo_grid_safe_s2") == "RL – hálózatbarát"
    assert F.base("ppo_pv_max_s0") == "ppo_pv_max" and F.base("oracle") == "oracle"
    assert F.PROFILE_HU["kiegyensúlyozott" and "balanced"] == "kiegyensúlyozott"


def test_all_figures_and_table_are_written(tmp_path):
    kp_std, kp_hard = _kpis(0), _kpis(1)
    eps = _episodes()
    lines = [f"L_x{i}" for i in range(6)]
    sens = []
    for axis, (_, values) in SENS_AXES.items():
        for v in values:
            for pol in ("uncontrolled", "line_following", "oracle", "ppo_grid_safe_s0", "ppo_grid_safe_s1"):
                gain = {"uncontrolled": 0.0, "oracle": 0.6}.get(pol, 0.3)
                for d in (1, 2):
                    sens.append(dict(axis=axis, value=v, policy=pol, day=d, ev_sessions=10 + v if axis == "n" else 12, ev_pv_share=0.3 + 0.1 * gain, ev_line_overload_kvah=40 * (1 - gain)))
    sessions = pd.DataFrame(
        dict(
            day_type=["workday"] * 50 + ["weekend"] * 50,
            start_hour=np.random.rand(100) * 24,
            energy_kwh=np.random.rand(100) * 40 + 2,
            stay_h=np.random.rand(100) * 10 + 0.5,
        )
    )
    curves = {"workday": np.random.rand(30, 24) * 50, "weekend": np.random.rand(30, 24) * 20}
    paths = []
    paths += F.fig_network(load_campus_data(), tmp_path)
    paths += F.fig_gmm_sessions(sessions, dict(n_fit_workday=2466, n_fit_weekend=50), tmp_path)
    paths += F.fig_gmm_load(curves, tmp_path)
    paths += F.fig_campus(eps, lines, 250.0, tmp_path, trafo_idx=2)
    paths += F.fig_rl_profiles(kp_std, tmp_path)
    paths += F.fig_example_day(eps, lines, tmp_path)
    paths += F.fig_battery_trafo(kp_hard, eps, 250.0, tmp_path, trafo_idx=2)
    paths += F.fig_sensitivity(pd.DataFrame(sens), tmp_path)
    assert len(paths) == 16 and all(p.exists() and p.stat().st_size > 5000 for p in paths)
    assert {p.suffix for p in paths} == {".png", ".pdf"}
    table = F.result_table({"standard": kp_std, "hard": kp_hard}, tmp_path)
    df = pd.read_csv(table, encoding="utf-8-sig")
    assert {"Eset", "Stratégia", "Vezetéktúlterhelés (kVAh/nap)"} <= set(df.columns)
    assert "Elméleti optimum (előrelátó)" in set(df["Stratégia"])
    assert (tmp_path / "tablazat.tex").exists()


def test_seed_stats_pools_the_seeds():
    st = F.seed_stats(_kpis(), ["ev_pv_share"])
    assert "ppo_pv_max" in st.index and "ppo_pv_max_s0" not in st.index
    assert (st[("std", "ev_pv_share")] >= 0).all()


@pytest.mark.skipif(not WORKDAY_FILE.exists(), reason="campus .gmm files not present")
def test_gmm_sampling_stage():
    sess, curves, info = gmm_samples(n_work=6, n_weekend=4, seed=1)
    assert set(sess["day_type"]) <= {"workday", "weekend"} and len(sess) > 0
    assert curves["workday"].shape == (6, 24) and curves["weekend"].shape == (4, 24)
    assert (curves["workday"] >= 0).all() and curves["workday"].max() <= 9 * 22.0 + 1e-6
    assert info["n_fit_workday"] > 0
