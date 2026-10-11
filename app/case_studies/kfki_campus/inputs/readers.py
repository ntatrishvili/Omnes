"""Readers for the raw input files. Each returns tidy DataFrames on the common clock."""

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.inputs.timebase import (
    LIVEVIEW_UTC_OFFSET_H,
    wall_to_cet,
)

# ---------------------------------------------------------------- LiveView

_DATA_ROW = re.compile(r"\d{4}-\d{2}-\d{2}")
_COLUMNS = {
    "Date/Time": "t",
    "kWh": "kwh",
    "kVArh": "kvarh",
    "Max kVA": "max_kva",  # daily export
    "PF (at max kVA)": "pf",  # daily export
    "kVA": "max_kva",  # interval export
    "PF": "pf",  # interval export
}


@dataclass
class LiveViewMeter:
    """One meter; may hold a daily history, interval data (30-min), or both.

    Frames have columns kwh, kvarh, max_kva, pf. ``daily`` is indexed by the LiveView
    calendar date, ``interval`` by interval start on fixed UTC+1.
    """

    meter_id: str
    title: str
    daily: pd.DataFrame | None = None
    interval: pd.DataFrame | None = None
    step: pd.Timedelta | None = None  # interval length
    ecode: str = ""  # 'E07' etc., how the Multiview export names the meter

    @property
    def resolution(self) -> str:
        has_d, has_i = self.daily is not None, self.interval is not None
        return "daily+interval" if has_d and has_i else "interval" if has_i else "daily"

    @property
    def data(self) -> pd.DataFrame:
        """The longest-history frame (daily if present, else interval)."""
        return self.daily if self.daily is not None else self.interval


def _interval_index(
    stamps: pd.DatetimeIndex, step: pd.Timedelta, utc_offset_h: int
) -> pd.DatetimeIndex:
    """End-labelled portal stamps (fixed ``utc_offset_h``) -> interval start, fixed UTC+1."""
    return stamps - step - pd.Timedelta(hours=utc_offset_h - 1)


def read_liveview(
    path: Path, utc_offset_h: int = LIVEVIEW_UTC_OFFSET_H
) -> LiveViewMeter:
    """Read one UtilityBook CSV (title line, header, data rows, footer line).

    Daily files are indexed by the portal's calendar date. Interval files (30-min,
    stamps are interval *ends* on the portal's fixed clock) are converted to interval
    starts on fixed UTC+1; the daily and interval exports agree day by day.
    """
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    title = lines[0]
    m = re.search(r"\((\d{6,})\)", title)
    if not m:
        raise ValueError(f"{path}: no meter id in title line {title!r}")
    e = re.search(r"\[(E\d+)\]", title)
    body = [lines[1]] + [ln for ln in lines[2:] if _DATA_ROW.match(ln)]
    df = pd.read_csv(io.StringIO("\n".join(body))).rename(columns=_COLUMNS)
    df["t"] = pd.to_datetime(df["t"])
    df = (
        df.set_index("t")
        .sort_index()
        .reindex(columns=["kwh", "kvarh", "max_kva", "pf"])
    )
    df = df[~df.index.duplicated(keep="last")]
    step = df.index.to_series().diff().median()
    ecode = e.group(1) if e else ""
    if step >= pd.Timedelta("1D"):
        return LiveViewMeter(m.group(1), title, daily=df, ecode=ecode)
    df.index = _interval_index(df.index, step, utc_offset_h)
    return LiveViewMeter(m.group(1), title, interval=df, step=step, ecode=ecode)


def read_multiview(
    path: Path,
    ecode_to_meter: dict[str, str],
    utc_offset_h: int = LIVEVIEW_UTC_OFFSET_H,
) -> dict[str, LiveViewMeter]:
    """LiveView 'Multiview' export: several meters side by side (active energy only).

    Header lines: ``Multiview`` / ``Date/Time,,kWh,kWh,..`` / ``,,<name> - P,<name> - P,..``.
    Meters are named by their E-code (``[E01]``), mapped to serials via the UtilityBook titles.
    """
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    names = next(ln for ln in lines[:6] if "[E" in ln).split(",")
    rows = [ln for ln in lines if _DATA_ROW.match(ln)]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    t = pd.to_datetime(df[0])
    step = t.diff().median()
    out = {}
    for j, name in enumerate(names):
        e = re.search(r"\[(E\d+)\]", name)
        if not e or e.group(1) not in ecode_to_meter:
            continue
        kwh = pd.to_numeric(df[j], errors="coerce")
        data = pd.DataFrame(
            {"kwh": kwh.to_numpy(), "kvarh": np.nan, "max_kva": np.nan, "pf": np.nan},
            index=_interval_index(pd.DatetimeIndex(t), step, utc_offset_h),
        )
        mid = ecode_to_meter[e.group(1)]
        out[mid] = LiveViewMeter(
            mid, name.strip(), interval=data, step=step, ecode=e.group(1)
        )
    return out


def _combine(a: pd.DataFrame | None, b: pd.DataFrame | None) -> pd.DataFrame | None:
    if a is None or b is None:
        return b if a is None else a
    return b.combine_first(a)  # b wins where both have a value


def _merge(old: LiveViewMeter, new: LiveViewMeter) -> LiveViewMeter:
    """Combine two sources of the same meter; daily and interval data are both kept."""
    return LiveViewMeter(
        old.meter_id,
        old.title,
        daily=_combine(old.daily, new.daily),
        interval=_combine(old.interval, new.interval),
        step=old.step or new.step,
        ecode=old.ecode or new.ecode,
    )


def read_liveview_dir(
    folder: Path, utc_offset_h: int = LIVEVIEW_UTC_OFFSET_H
) -> dict[str, LiveViewMeter]:
    """All UtilityBook and Multiview files in a folder, keyed by meter serial."""
    out: dict[str, LiveViewMeter] = {}
    for f in sorted(Path(folder).glob("UtilityBook_CSV_*.csv")):
        meter = read_liveview(f, utc_offset_h)
        out[meter.meter_id] = (
            _merge(out[meter.meter_id], meter) if meter.meter_id in out else meter
        )
    ecodes = {m.ecode: mid for mid, m in out.items() if m.ecode}
    for f in sorted(Path(folder).glob("Multiview_CSV_*.csv")):
        for mid, meter in read_multiview(f, ecodes, utc_offset_h).items():
            out[mid] = _merge(out[mid], meter) if mid in out else meter
    return out


# ------------------------------------------------------------------- E.ON


def _parse_eon_time(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    a = pd.to_datetime(s, format="%Y-%m-%d_%H:%M:%S", errors="coerce")
    b = pd.to_datetime(s, format="%Y.%m.%d. %H:%M", errors="coerce")
    return a.fillna(b)


def read_eon(paths: list[Path]) -> pd.DataFrame:
    """E.ON campus supply, sheet ``Adatok``, as 15-min energy on the common clock.

    The sheet is several appended blocks: two timestamp formats, and the status
    columns ('W' measured / 'E' estimated) vanish in later blocks, which shifts the
    reactive column. Returns columns kwh, kvarh (per 15 min) and ``estimated``
    (True where E.ON marked the value 'E').
    """
    frames = []
    for p in paths:
        raw = pd.read_excel(p, sheet_name="Adatok", header=0)
        raw.columns = [f"c{i}" for i in range(raw.shape[1])]
        for i in range(raw.shape[1], 7):
            raw[f"c{i}"] = np.nan
        t = _parse_eon_time(raw["c0"])
        kwh = pd.to_numeric(raw["c1"], errors="coerce")
        c2 = pd.to_numeric(raw["c2"], errors="coerce")  # numeric => no status column
        kvarh = c2.where(c2.notna(), pd.to_numeric(raw["c3"], errors="coerce"))
        # the export writes 0 for "no data" (the campus never draws 0 kWh): treat as missing
        invalid = ~(kwh > 0)
        kwh, kvarh = kwh.mask(invalid), kvarh.mask(invalid)
        estimated = raw["c2"].astype(str).str.strip().eq("E")
        frames.append(
            pd.DataFrame(
                {"t": t, "kwh": kwh, "kvarh": kvarh, "estimated": estimated}
            ).dropna(subset=["t"])
        )
    d = pd.concat(frames).drop_duplicates("t", keep="last").sort_values("t")
    d["t"] = wall_to_cet(pd.DatetimeIndex(d["t"]))
    d = d.dropna(subset=["t"]).drop_duplicates("t", keep="last").set_index("t")
    return d


def eon_hourly(eon15: pd.DataFrame) -> pd.DataFrame:
    """Hourly mean power: kw (= kWh per hour), kvar, and ``n`` samples per hour.

    Hours with fewer than four quarter-hours are returned as NaN (to be filled and
    flagged by the caller).
    """
    g = eon15.resample("1h")
    out = pd.DataFrame(
        {
            "kw": g["kwh"].sum(min_count=4),
            "kvar": g["kvarh"].sum(min_count=4),
            "n": g["kwh"].count(),
            "estimated": g["estimated"].max().fillna(False).astype(bool),
        }
    )
    return out


# ----------------------------------------------------------------- weather

SENTINEL_TEMP_STD = 5.0  # a 1-min mean with std > 5 degC is a logger glitch


def read_weather(paths: list[Path]) -> pd.DataFrame:
    """1-min logger files -> cleaned 1-min frame (fixed UTC+1, no DST).

    Drops the unit rows of the 2026 file and duplicate records, and replaces
    sentinel / implausible values by NaN (never zero).
    """
    frames = []
    for p in paths:
        d = pd.read_csv(p, low_memory=False)
        d["TIMESTAMP"] = pd.to_datetime(
            d["TIMESTAMP"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        frames.append(d.dropna(subset=["TIMESTAMP"]))
    d = pd.concat(frames).drop_duplicates("TIMESTAMP", keep="first")
    d = d.set_index("TIMESTAMP").sort_index()
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    bad_t = (
        (d["Temp_Avg"] < -30)
        | (d["Temp_Avg"] > 50)
        | (d["Temp_Std"] > SENTINEL_TEMP_STD)
    )
    d.loc[bad_t, ["Temp_Avg", "Temp_Std"]] = np.nan
    d.loc[d["Hum_Avg"] < 0, "Hum_Avg"] = np.nan
    d.loc[(d["Pressure_Avg"] < 900) | (d["Pressure_Avg"] > 1100), "Pressure_Avg"] = (
        np.nan
    )
    d["IrradGlobal_Avg"] = d["IrradGlobal_Avg"].clip(lower=0, upper=1400)
    return d


def weather_hourly(w: pd.DataFrame, max_gap_h: int = 2) -> pd.DataFrame:
    """Hourly means (timestamp = start of hour) of GHI, air temp, wind (m/s), pressure.

    NaN gaps up to ``max_gap_h`` hours are interpolated; the column ``interpolated``
    marks hours that had no valid 1-min sample at all.
    """
    g = w.resample("1h")
    h = pd.DataFrame(
        {
            "ghi": g["IrradGlobal_Avg"].mean(),
            "temp_c": g["Temp_Avg"].mean(),
            "wind_ms": g["WindSpeed_Avg"].mean() / 3.6,  # km/h -> m/s
            "pressure_pa": g["Pressure_Avg"].mean() * 100.0,  # mbar -> Pa
        }
    )
    h["interpolated"] = h[["ghi", "temp_c"]].isna().any(axis=1)
    h = h.interpolate(limit=max_gap_h, limit_area="inside")
    h.loc[h["ghi"] < 1.0, "ghi"] = 0.0  # logger night offset
    return h
