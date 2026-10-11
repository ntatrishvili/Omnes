"""Assemble the hourly input CSV of the case study from the raw files.

  python -m app.case_studies.kfki_campus.inputs.build_inputs --start 2025-05-28 --end 2026-03-01

Outputs (in ``data/processed``):
  kfki_timeseries_<start>_<end>.csv   the wide CSV of the README contract (fixed UTC+1, hourly, kW)
  load_tan_phi.csv                    signed Q/P per load (from kVArh/kWh)
  imputation_flags.csv                hourly flags: campus_filled, campus_estimated, residual_clipped, weather_interpolated
  meter_download_priority.csv         unmetered loads ranked by rated kVA with candidate register meters
  data_quality_report.md, qc_*.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.case_studies.kfki_campus.excel_io import (
    DATA_DIR,
    load_campus_data,
)  # noqa: E402
from app.case_studies.kfki_campus.inputs.disaggregate import (  # noqa: E402
    daily_shape,
    fill_campus_gaps,
    interval_gap_hours,
    mapped_load_kw,
    meter_daily_kwh,
)
from app.case_studies.kfki_campus.inputs.heat import heat_columns  # noqa: E402
from app.case_studies.kfki_campus.inputs.pv import PVConfig, pv_per_kwp  # noqa: E402
from app.case_studies.kfki_campus.inputs.readers import (  # noqa: E402
    eon_hourly,
    read_eon,
    read_liveview_dir,
    read_weather,
    weather_hourly,
)
from app.case_studies.kfki_campus.inputs.register import (  # noqa: E402
    download_priority,
    read_register,
)
from app.case_studies.kfki_campus.inputs.residual import (
    allocate,
    residual_kw,
)  # noqa: E402
from app.case_studies.kfki_campus.inputs.timebase import (  # noqa: E402
    LIVEVIEW_UTC_OFFSET_H,
    hourly_index,
    local_date,
)
from app.case_studies.kfki_campus.inputs.ev import (  # noqa: E402
    read_ev_profiles,
    synthetic_ev_profiles,
)

INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
PEAK_REPORTED_KW = 2055.0  # worst hour, Dec 2023 (report p.15)


def build_inputs(
    start: str,
    end: str,
    input_dir: Path = INPUT_DIR,
    out_dir: Path = PROCESSED_DIR,
    pv_cfg: PVConfig = PVConfig(),
    ev_wall_clock: bool = True,
    utc_offset_h: int = LIVEVIEW_UTC_OFFSET_H,
) -> Path:
    """Run the whole pipeline and return the path of the wide CSV."""
    input_dir, out_dir = Path(input_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = hourly_index(start, end)
    data = load_campus_data()

    # --- raw sources
    meters = read_liveview_dir(input_dir, utc_offset_h)
    mapping = pd.read_excel(DATA_DIR / "meter_map.xlsx", sheet_name="map", dtype=str)
    missing = set(mapping["meter_id"]) - set(meters)
    if missing:
        raise FileNotFoundError(
            f"mapped meters without a UtilityBook file: {sorted(missing)}"
        )
    unknown = set(mapping["load_id"]) - set(data.loads["id"])
    if unknown:
        raise ValueError(f"meter_map refers to unknown loads: {sorted(unknown)}")
    eon_paths = sorted(input_dir.glob("EON adatok kimutatas *.xlsx"))
    eon15 = read_eon(eon_paths)
    eon_h = eon_hourly(eon15)
    weather = weather_hourly(
        read_weather(sorted(input_dir.glob("meteorological_data_*.csv")))
    )

    # --- campus total, gaps filled using the metered daily energy
    wide = pd.date_range(eon_h.index.min(), eon_h.index.max(), freq="1h")
    campus_raw = eon_h["kw"].reindex(wide)
    metered_daily = pd.concat(
        [meter_daily_kwh(meters[m], utc_offset_h) for m in mapping["meter_id"]], axis=1
    ).sum(axis=1, min_count=1)
    campus, filled = fill_campus_gaps(
        campus_raw, metered_daily, utc_offset_h=utc_offset_h
    )
    shape = daily_shape(campus, utc_offset_h)

    # --- metered buildings, residual, allocation
    metered_kw, tan_phi = mapped_load_kw(mapping, meters, shape, utc_offset_h)
    metered_kw = metered_kw.reindex(idx)
    campus_i = campus.reindex(idx)
    resid, clipped = residual_kw(campus_i, metered_kw.sum(axis=1, min_count=1))
    unmetered = data.loads[~data.loads["id"].isin(mapping["load_id"])]
    resid_kw = allocate(resid, unmetered)

    out = pd.DataFrame(index=idx)
    for load in data.loads.itertuples():
        src = metered_kw[load.id] if load.id in metered_kw else resid_kw[load.id]
        out[load.csv_column] = src
    for hp in data.heat_pumps.itertuples():
        out[hp.csv_column] = 0.0  # inside the building meters / residual
    if data.has_heat_side:  # synthetic thermal demand for heat-coupled scenarios
        out = out.join(heat_columns(weather["temp_c"].reindex(idx), data))
    pv =pv_per_kwp(weather.reindex(idx), pv_cfg)
    out["pv_per_kwp"] = pv
    ev_paths = sorted(input_dir.glob("simulated_profile_*.csv"))
    ev_synthetic = len(ev_paths) == 0
    if ev_synthetic:
        ev = synthetic_ev_profiles(idx, len(data.ev_chargers))
        ev_zeroed = 0
    else:
        ev, ev_zeroed = read_ev_profiles(ev_paths, idx, ev_wall_clock)
    if len(ev.columns) < len(data.ev_chargers):
        raise ValueError(
            f"{len(ev.columns)} EV profile files for {len(data.ev_chargers)} chargers"
        )
    out = pd.concat([out, ev], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    # LiveView days run 1 h ahead of CET, so the last CET hour(s) of the window can belong
    # to a day without E.ON data: forward-fill at most 2 such edge hours (flagged).
    edge = out.isna().any(axis=1)
    if 0 < edge.sum() <= 2 and edge.iloc[-int(edge.sum()) :].all():
        out = out.ffill()
    else:
        edge = edge & False
    nan_cols = out.columns[out.isna().any()].tolist()
    if nan_cols:
        bad = out.index[out.isna().any(axis=1)]
        raise ValueError(
            f"NaN left in output columns {nan_cols}: {len(bad)} hours, "
            f"first {bad[:3].tolist()}, last {bad[-3:].tolist()}"
        )
    missing_cols = set(data.csv_columns) - set(out.columns)
    if missing_cols:
        raise ValueError(f"contract columns missing: {sorted(missing_cols)}")

    # --- write
    stem = f"kfki_timeseries_{start}_{end}"
    csv_path = out_dir / f"{stem}.csv"
    out.rename_axis("timestamp").to_csv(csv_path, date_format="%Y-%m-%d %H:%M:%S")
    # hourly weather next to the CSV (used by the figures, not by the simulation)
    weather.reindex(idx)[["ghi", "temp_c", "wind_ms"]].rename_axis("timestamp").to_csv(
        out_dir / "weather_hourly.csv", date_format="%Y-%m-%d %H:%M:%S"
    )

    eon_tan = float(eon_h["kvar"].sum() / eon_h["kw"].sum())
    tan = {
        l: (tan_phi[l] if np.isfinite(tan_phi.get(l, np.nan)) else eon_tan)
        for l in data.loads["id"]
    }
    pd.Series(tan, name="tan_phi").rename_axis("load_id").to_csv(
        out_dir / "load_tan_phi.csv"
    )

    flags = pd.DataFrame(
        {
            "campus_filled": filled.reindex(idx),
            "campus_estimated_by_eon": eon_h["estimated"].reindex(idx).fillna(False),
            "residual_clipped": clipped,
            "meter_gap_scaled": pd.concat(
                [interval_gap_hours(meters[m], idx) for m in set(mapping["meter_id"])],
                axis=1,
            ).any(axis=1),
            "edge_filled": edge,
            "weather_interpolated": weather["interpolated"].reindex(idx),
        }
    ).fillna(False)
    flags.rename_axis("timestamp").to_csv(out_dir / "imputation_flags.csv")

    prio = download_priority(
        data.loads, read_register(_register_path(input_dir)), set(mapping["load_id"])
    )
    prio.to_csv(out_dir / "meter_download_priority.csv", index=False)

    _qc_figures(out_dir, idx, campus_i, metered_kw, resid, out, weather.reindex(idx))
    _report(
        out_dir / "data_quality_report.md",
        start,
        end,
        idx,
        out,
        flags,
        campus_i,
        metered_kw,
        resid,
        pv,
        tan,
        ev_synthetic,
        mapping,
        pv_cfg,
        meters,
        utc_offset_h,
    )
    return csv_path


def _register_path(input_dir: Path) -> Path:
    found = sorted(input_dir.glob("KFKI_merok_kapcsolatai*.xlsx"))
    if not found:
        raise FileNotFoundError(f"meter register not found in {input_dir}")
    return found[-1]


def _qc_figures(out_dir, idx, campus, metered, resid, out, weather) -> None:
    day = lambda s: s.groupby(local_date(s.index)).sum() / 1000.0  # noqa: E731  MWh/day
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(day(campus), color="#2a78d6", lw=1.3, label="E.ON campus total")
    ax.plot(
        day(metered.sum(axis=1)), color="#eb6834", lw=1.3, label="13 metered points"
    )
    ax.plot(day(resid), color="#898781", lw=1.3, label="unmetered residual")
    ax.set_ylabel("MWh per day")
    ax.set_title("Daily energy balance of the input data", loc="left")
    ax.legend(fontsize=8, ncol=3)
    fig.savefig(out_dir / "qc_energy_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    daily_ghi = weather["ghi"].groupby(weather.index.normalize()).sum() / 1000.0
    daily_pv = out["pv_per_kwp"].groupby(out.index.normalize()).sum()
    a1.scatter(daily_ghi, daily_pv, s=8, color="#2a78d6")
    a1.set_xlabel("daily GHI (kWh/m2)")
    a1.set_ylabel("PV yield (kWh/kWp/day)")
    a1.set_title("PV model vs measured GHI", loc="left")
    wk = out.loc[(out.index >= "2025-07-07") & (out.index < "2025-07-14"), "pv_per_kwp"]
    if len(wk):
        a2.plot(wk.index, wk, color="#eb6834", lw=1.3)
        a2.set_ylabel("kW per kWp")
        a2.set_title("one summer week", loc="left")
        a2.tick_params(axis="x", labelrotation=30)
    fig.savefig(out_dir / "qc_pv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _interval_coverage(meters, mapping, idx) -> str:
    parts = []
    for m in sorted(set(mapping["meter_id"])):
        iv = meters[m].interval
        n = (
            0
            if iv is None
            else int(iv["kwh"].dropna().index.floor("1h").unique().isin(idx).sum())
        )
        parts.append(f"{m}: {n}/{len(idx)}")
    return ", ".join(parts)


def _ev_line(out: pd.DataFrame, synthetic: bool) -> str:
    cols = [c for c in out.columns if c.startswith("ev_charger_")]
    ev = out[cols]
    src = (
        "**synthetic placeholders**"
        if synthetic
        else "user data (simulated_profile_*.csv, read as local wall-clock time)"
    )
    return (
        f"EV: {len(cols)} charger profiles, {src}; per charger mean "
        f"{ev.mean().mean():.2f} kW, hourly peak up to {ev.max().max():.0f} kW, "
        f"{ev.sum().mean() / 1000:.1f} MWh each over the period."
    )


def _md_table(df: pd.DataFrame) -> str:
    head = "| month | " + " | ".join(df.columns) + " |"
    sep = "|---|" + "---|" * len(df.columns)
    rows = [
        f"| {i:%Y-%m} | " + " | ".join(f"{v:,.0f}" for v in r) + " |"
        for i, r in df.iterrows()
    ]
    return "\n".join([head, sep, *rows])


def _report(
    path,
    start,
    end,
    idx,
    out,
    flags,
    campus,
    metered,
    resid,
    pv,
    tan,
    ev_syn,
    mapping,
    cfg,
    meters,
    utc_offset_h,
):
    n = len(idx)
    m_tot = metered.sum(axis=1)
    mon = (
        pd.DataFrame(
            {"campus_MWh": campus, "metered_MWh": m_tot, "residual_MWh": resid}
        )
        .resample("MS")
        .sum()
        / 1000.0
    )
    lines = [
        f"# Input data quality report ({start} -> {end})",
        "",
        f"* {n} hourly rows ({n // 24} days), clock = fixed UTC+1, power in kW",
        f"* campus peak (hourly): **{campus.max():.0f} kW** (reported worst hour Dec 2023: {PEAK_REPORTED_KW:.0f} kW); mean {campus.mean():.0f} kW",
        f"* metered points: {mapping['meter_id'].nunique()} meters -> {mapping['load_id'].nunique()} model loads, "
        f"{m_tot.sum() / campus.sum():.1%} of campus energy; unmetered residual {resid.sum() / campus.sum():.1%}",
        "",
        "## Imputation / flags (share of hours)",
        "",
        "| flag | hours | share |",
        "|---|---|---|",
    ]
    for c in flags.columns:
        lines.append(f"| {c} | {int(flags[c].sum())} | {flags[c].mean():.2%} |")
    lines += [
        "",
        "Metered energy is measured (no imputation). Measured 30-min hours in the window "
        f"per meter: {_interval_coverage(meters, mapping, idx)}. Hours without interval data "
        "use the daily energy and the campus shape.",
        f"LiveView clock assumed fixed UTC+{utc_offset_h}: the residual E.ON - metered is smoothest when the "
        "LiveView stamps are shifted back by 1 h (see README); confirm in the LiveView time-zone setting.",
        _ev_line(out, ev_syn),
        "Heat-pump electric columns are 0 (already inside building meters / residual). "
        "`heat_*` columns are **synthetic** thermal demand (degree-hour model on the "
        "measured outdoor temperature, scaled to the assumed heat-pump sizes).",
        "",
        "## Monthly energy (MWh)",
        "",
        _md_table(mon.round(0)),
        "",
        "## PV model",
        "",
        f"* site {cfg.latitude:.6f} N, {cfg.longitude:.6f} E, tilt {cfg.tilt:.0f} deg, azimuth {cfg.azimuth:.0f} deg (tilt/azimuth assumed)",
        f"* yield over the period: {pv.sum():.0f} kWh/kWp (no annual extrapolation: the period has no spring); peak {pv.max():.2f} kW/kWp",
        f"* the 9 kWp system would peak at {9 * pv.max():.1f} kW (measured reading in the report: 7.41 kW)",
        "",
        "## Signed tan(phi) per load (Q/P, capacitive < 0)",
        "",
        ", ".join(f"{k}: {v:+.2f}" for k, v in sorted(tan.items())),
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--start", default="2025-05-28")
    p.add_argument("--end", default="2026-03-01", help="last day, inclusive")
    p.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument(
        "--ev-clock",
        choices=["local", "fixed"],
        default="local",
        help="clock of the EV profile timestamps: Budapest wall time (default) or already fixed UTC+1",
    )
    p.add_argument("--tilt", type=float, default=PVConfig.tilt)
    p.add_argument("--azimuth", type=float, default=PVConfig.azimuth)
    p.add_argument(
        "--liveview-offset",
        type=int,
        default=LIVEVIEW_UTC_OFFSET_H,
        help="fixed UTC offset (h) of the LiveView portal clock",
    )
    a = p.parse_args(argv)
    cfg = PVConfig(tilt=a.tilt, azimuth=a.azimuth)
    path = build_inputs(
        a.start,
        a.end,
        a.input_dir,
        a.out_dir,
        cfg,
        a.ev_clock == "local",
        a.liveview_offset,
    )
    print("written:", path)


if __name__ == "__main__":
    main()
