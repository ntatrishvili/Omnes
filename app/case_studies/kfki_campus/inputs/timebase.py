"""Common clock handling.

The weather logger runs on fixed UTC+1. The E.ON export and LiveView use Budapest
wall-clock time (CET/CEST). Everything is converted to naive timestamps that mean
"fixed UTC+1", so the simulation year has no repeated or missing hour.
"""

import numpy as np
import pandas as pd

CET_FIXED = "Etc/GMT-1"  # note the POSIX sign: this is UTC+1
WALL = "Europe/Budapest"


def wall_to_cet(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Budapest wall-clock stamps -> naive fixed-UTC+1 stamps.

    Non-existent stamps (spring gap) become NaT. Ambiguous stamps (autumn hour) are
    taken as the first (summer-time) pass; the exports carry that hour only once.
    """
    loc = index.tz_localize(
        WALL, ambiguous=np.ones(len(index), dtype=bool), nonexistent="NaT"
    )
    return loc.tz_convert(CET_FIXED).tz_localize(None)


def local_date(index_cet: pd.DatetimeIndex) -> pd.Index:
    """Budapest calendar date of each fixed-UTC+1 stamp (days have 23/24/25 hours)."""
    loc = index_cet.tz_localize(CET_FIXED).tz_convert(WALL)
    return pd.Index(loc.tz_localize(None).normalize())


LIVEVIEW_UTC_OFFSET_H = 2  # LiveView portal clock: fixed UTC+2 (estimated, see README)


def lv_day(
    index_cet: pd.DatetimeIndex, utc_offset_h: int = LIVEVIEW_UTC_OFFSET_H
) -> pd.Index:
    """LiveView calendar date of each fixed-UTC+1 hour (its days run on the portal clock)."""
    shift = pd.Timedelta(hours=utc_offset_h - 1)
    return pd.Index((index_cet + shift).normalize())


def hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    """Hourly fixed-UTC+1 index from ``start`` 00:00 to ``end`` 23:00 inclusive."""
    return pd.date_range(
        pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23), freq="1h"
    )
