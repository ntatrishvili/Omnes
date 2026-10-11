"""EV charger demand profiles (``simulated_profile_*.csv``): one file per charger."""

from pathlib import Path

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.inputs.timebase import wall_to_cet


def ev_column(k: int) -> str:
    """CSV column of charger ``k`` (1-based), matching the ev_chargers sheet."""
    return f"ev_charger_{k:02d}"


def read_ev_profiles(
    paths: list[Path], index: pd.DatetimeIndex, wall_clock: bool = True
) -> tuple[pd.DataFrame, int]:
    """Hourly kW per charger on ``index`` (fixed UTC+1), columns ev_charger_01...

    Files are taken in sorted (seed) order; the profiles are 15-min mean power in kW.
    The timestamps come from a behavioural simulation and are read as local wall-clock
    time (Europe/Budapest, with daylight saving) unless ``wall_clock`` is False.
    Returns the table and the number of hours that had to be set to 0 (spring DST gap).
    """
    cols, filled = {}, 0
    for k, p in enumerate(sorted(paths), start=1):
        d = pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp")
        s = d["total_power_kw"].astype(float)
        if wall_clock:
            s.index = wall_to_cet(pd.DatetimeIndex(s.index))
            s = s[s.index.notna()]
        h = s.resample("1h").mean().reindex(index)
        h = h.interpolate(limit=2, limit_area="inside")
        filled += int(h.isna().sum())
        cols[ev_column(k)] = h.fillna(0.0)
    return pd.DataFrame(cols, index=index), filled


def synthetic_ev_profiles(index: pd.DatetimeIndex, n: int = 10) -> pd.DataFrame:
    """Placeholder when no profile files exist: the weekday-peak shape, 10 % apart."""
    from app.case_studies.kfki_campus.synthetic import ev_per_charger_profile

    base = ev_per_charger_profile(index)
    return pd.DataFrame(
        {ev_column(k): base * (0.8 + 0.04 * k) for k in range(1, n + 1)}, index=index
    )
