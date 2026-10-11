import numpy as np
import pandas as pd
import pytest

from app.case_studies.kfki_campus.excel_io import DATA_DIR, load_campus_data
from app.case_studies.kfki_campus.inputs.build_inputs import INPUT_DIR
from app.case_studies.kfki_campus.inputs.disaggregate import (
    daily_shape,
    fill_campus_gaps,
    interval_gap_hours,
    interval_hourly_kw,
    meter_kw,
)
from app.case_studies.kfki_campus.inputs.ev import (
    read_ev_profiles,
    synthetic_ev_profiles,
)
from app.case_studies.kfki_campus.inputs.pv import PVConfig, pv_per_kwp
from app.case_studies.kfki_campus.inputs.readers import (
    LiveViewMeter,
    eon_hourly,
    read_eon,
    read_liveview,
    read_liveview_dir,
    read_multiview,
    read_weather,
    weather_hourly,
)
from app.case_studies.kfki_campus.inputs.residual import allocate, residual_kw
from app.case_studies.kfki_campus.inputs.timebase import (
    hourly_index,
    local_date,
    lv_day,
    wall_to_cet,
)


def test_wall_to_cet_summer_winter_and_spring_gap():
    idx = pd.DatetimeIndex(
        ["2025-07-01 12:00", "2025-01-15 12:00", "2025-03-30 02:30", "2025-03-30 03:00"]
    )
    out = wall_to_cet(idx)
    assert out[0] == pd.Timestamp("2025-07-01 11:00")  # CEST = UTC+2 -> fixed UTC+1
    assert out[1] == pd.Timestamp("2025-01-15 12:00")
    assert pd.isna(out[2])  # hour that does not exist
    assert out[3] == pd.Timestamp("2025-03-30 02:00")


def test_local_date_follows_budapest_day():
    # 23:00 fixed UTC+1 in summer is already 00:00 of the next local day
    idx = pd.DatetimeIndex(["2025-07-01 22:00", "2025-07-01 23:00"])
    assert list(local_date(idx)) == [
        pd.Timestamp("2025-07-01"),
        pd.Timestamp("2025-07-02"),
    ]


def test_read_liveview_daily(tmp_path):
    f = tmp_path / "UtilityBook_CSV_36708390_x.csv"
    f.write_text(
        "KFKI - [E13] 17. Epulet (36708390) []\n"
        "Date/Time,Normalised,kWh,kVArh,Max kVA,PF (at max kVA)\n"
        "2025-05-01,,,,,\n"
        "2025-05-28,,1000.5,250.0,60.0,0.97\n"
        "2025-05-29,,1100.0,-10.0,61.0,0.96\n"
        "Powered by liveview. Copyright (c) Livewire\n",
        encoding="utf-8",
    )
    m = read_liveview(f)
    assert m.meter_id == "36708390" and m.resolution == "daily"
    assert m.data["kwh"].loc["2025-05-28"] == 1000.5
    assert np.isnan(m.data["kwh"].loc["2025-05-01"])
    assert m.data["kvarh"].loc["2025-05-29"] == -10.0  # capacitive stays negative


def _interval_file(path, meter="36708419", ecode="E07", offset_rows=None):
    rows = offset_rows or [
        # end-labelled 30-min stamps on the portal clock (UTC+2): first interval 00:00-00:30
        ("2025-06-10 00:30", 5.0, 1.0, 20.0, 0.97),
        ("2025-06-10 01:00", 6.0, 1.5, 21.0, 0.96),
        ("2025-06-10 01:30", None, None, None, None),
        ("2025-06-10 02:00", 8.0, 2.0, 22.0, 0.95),
    ]
    body = "\n".join(
        f"{t},,{'' if k is None else k},{'' if q is None else q},{'' if v is None else v},{'' if p is None else p}"
        for t, k, q, v, p in rows
    )
    path.write_text(
        f"KFKI - [{ecode}] 3. Epulet ({meter}) []\nDate/Time,Normalised,kWh,kVArh,kVA,PF\n"
        f"{body}\nPowered by liveview. Copyright (c) Livewire\n",
        encoding="utf-8",
    )
    return path


def test_read_liveview_interval_converts_end_label_and_clock(tmp_path):
    m = read_liveview(_interval_file(tmp_path / "UtilityBook_CSV_a.csv"))
    assert m.resolution == "interval" and m.step == pd.Timedelta("30min")
    assert m.ecode == "E07"
    # stamp 00:30 (end, UTC+2) = interval start 00:00 UTC+2 = 23:00 the day before in UTC+1
    assert m.interval.index[0] == pd.Timestamp("2025-06-09 23:00")
    assert m.interval["kwh"].iloc[0] == 5.0
    assert m.interval["max_kva"].iloc[0] == 20.0  # 'kVA' column of the interval export


def test_read_multiview_and_merge_keeps_daily_history(tmp_path):
    (tmp_path / "UtilityBook_CSV_daily.csv").write_text(
        "KFKI - [E01] 1. Epulet (36708411) []\n"
        "Date/Time,Normalised,kWh,kVArh,Max kVA,PF (at max kVA)\n"
        "2025-06-10,,1800.0,-400.0,80.0,0.97\n"
        "2025-06-11,,1700.0,-380.0,79.0,0.97\n"
        "Powered by liveview. Copyright (c) Livewire\n",
        encoding="utf-8",
    )
    (tmp_path / "Multiview_CSV_x.csv").write_text(
        "Multiview\nDate/Time,,kWh,kWh\n"
        ",,KFKI - [E01] 1. Epulet - P,KFKI - [E99] other - P\n"
        "2025-06-12 00:30,,30.0,1.0\n"
        "2025-06-12 01:00,,31.0,1.0\n"
        "Powered by liveview. Copyright (c) Livewire\n",
        encoding="utf-8",
    )
    meters = read_liveview_dir(tmp_path)
    assert set(meters) == {"36708411"}  # unknown E99 ignored
    m = meters["36708411"]
    assert m.resolution == "daily+interval"
    assert len(m.daily) == 2  # the interval export did not replace the daily history
    assert m.interval["kwh"].tolist() == [30.0, 31.0]


def test_meter_kw_prefers_measured_interval_hours():
    idx = pd.date_range("2025-06-09 23:00", periods=48, freq="1h")
    shape = pd.Series(1 / 24, index=idx)
    interval = pd.DataFrame(
        {"kwh": [2.0, 3.0]},
        index=pd.DatetimeIndex(["2025-06-09 23:00", "2025-06-09 23:30"]),
    )
    meter = LiveViewMeter(
        "1",
        "t",
        daily=pd.DataFrame({"kwh": [2400.0]}, index=pd.DatetimeIndex(["2025-06-10"])),
        interval=interval,
        step=pd.Timedelta("30min"),
    )
    kw = meter_kw(meter, shape)
    assert kw.iloc[0] == 5.0  # measured hour (two half-hours) wins over 2400/24 = 100
    assert kw.iloc[1] == pytest.approx(
        100.0
    )  # rest of the LiveView day from the daily value


def test_incomplete_hour_is_scaled_not_dropped():
    idx = pd.date_range("2025-06-10 00:00", periods=4, freq="30min")
    kwh = pd.DataFrame(
        {"kwh": [5.0, 5.0, 6.0, None]}, index=idx
    )  # last half-hour missing
    meter = LiveViewMeter("1", "t", interval=kwh, step=pd.Timedelta("30min"))
    h = interval_hourly_kw(meter)
    assert h.iloc[0] == 10.0 and h.iloc[1] == 12.0  # 6 kWh in one half hour -> 12 kW
    gaps = interval_gap_hours(
        meter, pd.date_range("2025-06-10 00:00", periods=2, freq="1h")
    )
    assert gaps.tolist() == [False, True]


def test_read_ev_profiles_local_clock_and_hourly_mean(tmp_path):
    t = pd.date_range("2025-07-01 08:00", periods=8, freq="15min")  # wall time (CEST)
    for i, scale in enumerate((1.0, 2.0)):
        pd.DataFrame(
            {"timestamp": t, "total_power_kw": [10.0, 10.0, 10.0, 10.0, 0, 0, 0, 0]}
        ).assign(total_power_kw=lambda d: d.total_power_kw * scale).to_csv(
            tmp_path / f"simulated_profile_seed{42 + i}.csv", index=False
        )
    idx = pd.date_range("2025-07-01 06:00", periods=4, freq="1h")  # fixed UTC+1
    ev, zeroed = read_ev_profiles(sorted(tmp_path.glob("simulated_profile_*.csv")), idx)
    assert list(ev.columns) == ["ev_charger_01", "ev_charger_02"]  # sorted file order
    # 08:00-09:00 CEST = 07:00-08:00 fixed UTC+1 -> mean 10 kW; hour after is 0
    assert ev["ev_charger_01"].loc["2025-07-01 07:00"] == pytest.approx(10.0)
    assert ev["ev_charger_02"].loc["2025-07-01 07:00"] == pytest.approx(20.0)
    assert ev["ev_charger_01"].loc["2025-07-01 08:00"] == 0.0
    ev_fixed, _ = read_ev_profiles(
        sorted(tmp_path.glob("simulated_profile_*.csv")), idx, wall_clock=False
    )
    assert ev_fixed["ev_charger_01"].loc["2025-07-01 08:00"] == pytest.approx(10.0)


def test_synthetic_ev_fallback_has_one_column_per_charger():
    idx = pd.date_range("2025-07-01", periods=48, freq="1h")
    ev = synthetic_ev_profiles(idx, 10)
    assert ev.shape == (48, 10) and ev.columns[0] == "ev_charger_01"
    assert ev.sum().is_monotonic_increasing  # the placeholders differ by charger


def test_lv_day_runs_one_hour_ahead_of_cet():
    idx = pd.DatetimeIndex(["2025-06-09 22:00", "2025-06-09 23:00"])
    assert list(lv_day(idx)) == [pd.Timestamp("2025-06-09"), pd.Timestamp("2025-06-10")]


def _eon_workbook(path):
    rows = [
        # block with status columns
        ("2025-06-23_00:00:00", 100.0, "W", 20.0, "W", 0, "W"),
        ("2025-06-23_00:15:00", 100.0, "W", 20.0, "W", 0, "W"),
        # later block: other timestamp format, no status columns, duplicates the 00:15 row
        ("2025.06.23. 00:15", 100.0, 20.0, 0, np.nan, np.nan, np.nan),
        ("2025.06.23. 00:30", 120.0, 25.0, 0, np.nan, np.nan, np.nan),
        ("2025.06.23. 00:45", 0.0, 0.0, 0, np.nan, np.nan, np.nan),  # 0 = no data
        ("2025.03.30. 02:15", 90.0, 10.0, 0, np.nan, np.nan, np.nan),  # DST gap
    ]
    df = pd.DataFrame(
        rows, columns=["Idő", "Érték", "Státusz", "Érték.1", "S1", "E2", "S2"]
    )
    with pd.ExcelWriter(path) as xw:
        df.to_excel(xw, sheet_name="Adatok", index=False)


def test_read_eon_formats_dedupe_zero_and_dst(tmp_path):
    f = tmp_path / "EON.xlsx"
    _eon_workbook(f)
    e = read_eon([f])
    # summer wall 00:00 -> 22:00 fixed UTC+1 of the previous day
    assert e.index.min() == pd.Timestamp("2025-06-22 23:00")
    assert len(e) == 4  # duplicate removed, DST-gap row dropped, zero row kept as NaN
    assert e["kwh"].isna().sum() == 1  # the 0.0 "no data" value
    assert (
        e.loc["2025-06-22 23:30", "kvarh"] == 25.0
    )  # status-less block, shifted column


def test_eon_hourly_requires_four_quarters():
    t = pd.date_range("2025-01-01", periods=6, freq="15min")
    e = pd.DataFrame({"kwh": 10.0, "kvarh": 1.0, "estimated": False}, index=t)
    h = eon_hourly(e)
    assert h["kw"].iloc[0] == 40.0 and np.isnan(h["kw"].iloc[1])


def test_read_weather_sentinels_and_unit_rows(tmp_path):
    f = tmp_path / "met.csv"
    cols = "TIMESTAMP,RECORD,IrradGlobal_Avg,Temp_Avg,Temp_Std,Hum_Avg,WindSpeed_Avg,Pressure_Avg"
    f.write_text(
        f"{cols}\nTS,RN,W/m2,oC,oC,%rH,km/h,mBar\n"
        "2026-01-01 00:00:00,1,0,-7.7,0.4,-100,7.2,1020\n"
        "2026-01-01 00:01:00,2,0,-42.2,44.7,-100,3.6,1055\n"
        "2026-01-01 00:01:00,3,0,-42.2,44.7,-100,3.6,1055\n"
        "2026-01-01 00:02:00,4,5,-7.9,0.2,50,3.6,1020\n",
        encoding="utf-8",
    )
    w = read_weather([f])
    assert len(w) == 3  # unit row and duplicate dropped
    assert np.isnan(w.loc["2026-01-01 00:01", "Temp_Avg"])  # glitch, not zero
    assert np.isnan(w.loc["2026-01-01 00:00", "Hum_Avg"])
    h = weather_hourly(w)
    assert h["temp_c"].iloc[0] == pytest.approx(-7.8)
    assert h["wind_ms"].iloc[0] == pytest.approx((7.2 + 3.6 + 3.6) / 3 / 3.6)


def _campus(days=30, start="2025-02-01"):
    idx = pd.date_range(start, periods=24 * days, freq="1h")
    rng = np.random.default_rng(1)
    return pd.Series(
        800 + 100 * np.sin(idx.hour / 24 * 2 * np.pi) + rng.normal(0, 5, len(idx)), idx
    )


def test_daily_shape_sums_to_one_per_local_day():
    c = _campus()
    s = daily_shape(c)
    full_days = s.groupby(lv_day(s.index)).sum()
    assert np.allclose(full_days.iloc[1:-1], 1.0)


def test_disaggregation_preserves_daily_energy():
    c = _campus()
    shape = daily_shape(c)
    days = lv_day(shape.index).unique()
    kwh = pd.Series(np.linspace(1000, 1500, len(days)), index=days)

    meter = LiveViewMeter("1", "t", daily=pd.DataFrame({"kwh": kwh}))
    kw = meter_kw(meter, shape)
    per_day = kw.groupby(lv_day(kw.index)).sum()
    assert np.allclose(per_day.iloc[1:-1], kwh.iloc[1:-1])


def test_fill_campus_gaps_uses_same_weekday_profile():
    c = _campus(days=60)
    full = c.copy()
    gap = c.index[(c.index >= "2025-02-20 06:00") & (c.index < "2025-02-20 12:00")]
    c[gap] = np.nan
    metered = pd.Series(1000.0, index=local_date(c.index).unique())
    filled, flag = fill_campus_gaps(c, metered)
    assert flag.sum() == len(gap) and not filled.isna().any()
    assert np.allclose(filled[gap], full[gap], rtol=0.15)


def test_residual_non_negative_and_allocation_sums():
    idx = pd.date_range("2025-01-01", periods=4, freq="1h")
    campus = pd.Series([900, 800, 500, 100.0], idx)
    metered = pd.Series([300, 300, 300, 300.0], idx)
    resid, clipped = residual_kw(campus, metered)
    assert (resid >= 0).all() and clipped.iloc[-1] and not clipped.iloc[0]
    loads = pd.DataFrame({"id": ["a", "b"], "rated_kVA": [100.0, 300.0]})
    alloc = allocate(resid, loads)
    assert np.allclose(alloc.sum(axis=1), resid)
    assert np.allclose(alloc["b"], 3 * alloc["a"])


def _weather(day):
    idx = pd.date_range(day, periods=24, freq="1h")
    ghi = np.clip(np.sin(np.pi * (idx.hour - 5.5) / 13), 0, None) * (
        900 if "-07-" in day else 300
    )
    return pd.DataFrame(
        {"ghi": ghi, "temp_c": 20.0, "wind_ms": 2.0, "pressure_pa": 101000.0}, index=idx
    )


def test_pv_profile_physical_bounds_and_seasonality():
    cfg = PVConfig()
    summer = pv_per_kwp(_weather("2025-07-01"), cfg)
    winter = pv_per_kwp(_weather("2025-12-21"), cfg)
    assert summer.iloc[[0, 1, 2, 22, 23]].eq(0).all()  # night
    assert 0 < summer.max() <= 1 / cfg.dc_ac_ratio + 1e-9
    assert summer.sum() > winter.sum()  # (fixture has the same daylight length)


@pytest.mark.skipif(
    not (INPUT_DIR / "meteorological_data_2025.csv").exists(),
    reason="raw input files not present",
)
def test_meter_map_covers_real_files_and_loads():
    mapping = pd.read_excel(DATA_DIR / "meter_map.xlsx", sheet_name="map", dtype=str)
    # the id in the title line is authoritative: the '21b' file is named after 36708392
    files = {read_liveview(p).meter_id for p in INPUT_DIR.glob("UtilityBook_CSV_*.csv")}
    ignored = pd.read_excel(
        DATA_DIR / "meter_map.xlsx", sheet_name="ignored", dtype=str
    )
    assert set(mapping["meter_id"]) | set(ignored["meter_id"]) == files
    assert set(mapping["load_id"]) <= set(load_campus_data().loads["id"])
    assert not mapping["meter_id"].duplicated().any()


def test_hourly_index_is_inclusive_of_last_day():
    idx = hourly_index("2025-05-28", "2025-05-29")
    assert len(idx) == 48 and idx[-1] == pd.Timestamp("2025-05-29 23:00")
