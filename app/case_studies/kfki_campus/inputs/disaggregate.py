"""Hourly building profiles from LiveView meters, and campus gap filling.

Where a meter has interval data (30-min, from LiveView interval/Multiview exports) it
is summed to hours directly. Everywhere else the meter's daily energy is spread over
the day with the campus' own normalised E.ON hourly shape of that day, so daily totals
are exact and day-to-day dynamics (weekends, holidays, weather) come from the real
campus. In those periods the intra-day shape is the campus', not the building's.

Days are LiveView days (the portal's fixed clock), not Budapest days.
"""

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.inputs.readers import LiveViewMeter
from app.case_studies.kfki_campus.inputs.timebase import LIVEVIEW_UTC_OFFSET_H, lv_day


def meter_daily_kwh(
    meter: LiveViewMeter, utc_offset_h: int = LIVEVIEW_UTC_OFFSET_H
) -> pd.Series:
    """Daily energy per LiveView date; days only covered by interval data are summed
    from it (complete days only)."""
    daily = (
        meter.daily["kwh"]
        if meter.daily is not None
        else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    )
    if meter.interval is None:
        return daily
    per_day = int(round(pd.Timedelta("1D") / meter.step))
    day = lv_day(meter.interval.index, utc_offset_h)
    g = meter.interval["kwh"].groupby(day)
    from_interval = g.sum(min_count=per_day).where(g.count() >= per_day)
    return daily.combine_first(from_interval)


def fill_campus_gaps(
    campus_kw: pd.Series,
    metered_daily_kwh: pd.Series,
    weeks: tuple[int, ...] = (1, 2, 3),
    utc_offset_h: int = LIVEVIEW_UTC_OFFSET_H,
) -> tuple[pd.Series, pd.Series]:
    """Fill NaN hours of the E.ON campus series; returns (filled series, filled flag).

    A missing hour is estimated from the same hour on the same weekday +-1..3 weeks
    (median), scaled by the ratio of the metered daily energy on the day to the same
    ratio-reference days' metered energy (clipped to 0.5..2).
    """
    out = campus_kw.copy()
    flag = campus_kw.isna()
    dates = lv_day(campus_kw.index, utc_offset_h)
    for ts in campus_kw.index[flag]:
        refs, ref_m = [], []
        for k in weeks:
            for sign in (-1, 1):
                t2 = ts + sign * k * pd.Timedelta(days=7)
                v = campus_kw.get(t2, np.nan)
                if pd.notna(v):
                    refs.append(v)
                    d2 = lv_day(pd.DatetimeIndex([t2]), utc_offset_h)[0]
                    ref_m.append(metered_daily_kwh.get(d2, np.nan))
        if not refs:
            continue
        scale = 1.0
        m_today = metered_daily_kwh.get(dates[campus_kw.index.get_loc(ts)], np.nan)
        m_ref = np.nanmedian(ref_m) if np.isfinite(ref_m).any() else np.nan
        if np.isfinite(m_today) and np.isfinite(m_ref) and m_ref > 0:
            scale = float(np.clip(m_today / m_ref, 0.5, 2.0))
        out[ts] = float(np.median(refs)) * scale
    return out, flag


def daily_shape(
    campus_kw: pd.Series, utc_offset_h: int = LIVEVIEW_UTC_OFFSET_H
) -> pd.Series:
    """Hour weights summing to 1 per LiveView day (NaN for incomplete days)."""
    d = lv_day(campus_kw.index, utc_offset_h)
    total = campus_kw.groupby(d).transform("sum")
    n = campus_kw.groupby(d).transform("count")
    w = campus_kw / total
    return w.where(n >= 24)


def meter_kw(
    meter: LiveViewMeter,
    shape: pd.Series,
    utc_offset_h: int = LIVEVIEW_UTC_OFFSET_H,
) -> pd.Series:
    """Hourly kW of one meter on the hourly index of ``shape``.

    Measured interval hours win; the rest comes from the daily energy x campus shape.
    """
    out = pd.Series(np.nan, index=shape.index)
    if meter.daily is not None:
        kwh_day = meter.daily["kwh"].reindex(lv_day(shape.index, utc_offset_h))
        out = shape * kwh_day.to_numpy()
    if meter.interval is not None:
        out = interval_hourly_kw(meter).reindex(shape.index).combine_first(out)
    return out


def interval_hourly_kw(meter: LiveViewMeter, max_gap_h: int = 3) -> pd.Series:
    """Hourly kW from interval data. An hour with a missing sub-interval is scaled up
    from the intervals present; gaps of up to ``max_gap_h`` hours are interpolated."""
    per_hour = int(round(pd.Timedelta("1h") / meter.step))
    g = meter.interval["kwh"].resample("1h")
    n = g.count()
    h = (g.sum(min_count=1) * per_hour / n.where(n > 0)).dropna()
    h = h.reindex(n.index)
    return h.interpolate(limit=max_gap_h, limit_area="inside")


def interval_gap_hours(meter: LiveViewMeter, index: pd.DatetimeIndex) -> pd.Series:
    """True for hours of ``index`` where this meter's interval data are incomplete."""
    if meter.interval is None:
        return pd.Series(False, index=index)
    per_hour = int(round(pd.Timedelta("1h") / meter.step))
    n = meter.interval["kwh"].resample("1h").count()
    first, last = meter.interval["kwh"].dropna().index[[0, -1]]
    covered = (index >= first.floor("1h")) & (index <= last.floor("1h"))
    return pd.Series(
        (n.reindex(index).fillna(0) < per_hour).to_numpy() & covered, index=index
    )


def mapped_load_kw(
    mapping: pd.DataFrame,
    meters: dict[str, LiveViewMeter],
    shape: pd.Series,
    utc_offset_h: int = LIVEVIEW_UTC_OFFSET_H,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Sum the mapped meters per model load.

    ``mapping`` has columns meter_id, load_id. Returns the hourly kW per load and the
    signed tan(phi) = sum(kvarh)/sum(kwh) per load (from each meter's longest-history
    frame, rows where both energies exist; NaN if no kVArh is available).
    """
    series, tan_phi = {}, {}
    for load_id, grp in mapping.groupby("load_id"):
        parts = [meter_kw(meters[m], shape, utc_offset_h) for m in grp["meter_id"]]
        series[load_id] = pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))
        kwh = kvarh = 0.0
        for m in grp["meter_id"]:
            both = meters[m].data[["kwh", "kvarh"]].dropna()
            kwh, kvarh = kwh + both["kwh"].sum(), kvarh + both["kvarh"].sum()
        tan_phi[load_id] = float(kvarh / kwh) if kwh > 0 else np.nan
    return pd.DataFrame(series), tan_phi
