"""Synthetic stand-in for the user-supplied time-series CSV (tests and smoke runs).

NOT real data: smooth diurnal/seasonal shapes scaled to the rated powers in the
Excel inputs, so that the pipeline can be exercised end to end.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.excel_io import CampusData, load_campus_data
from app.case_studies.kfki_campus.inputs.heat import heat_columns


def ev_per_charger_profile(idx: pd.DatetimeIndex) -> np.ndarray:
    """Placeholder demand of one charger in kW: morning and evening arrival peaks, weekdays."""
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    weekday = (idx.dayofweek.to_numpy() < 5).astype(float)
    ev = (
        np.exp(-0.5 * ((hour - 9) / 2) ** 2)
        + 0.6 * np.exp(-0.5 * ((hour - 17) / 2) ** 2)
    ) * weekday
    return 6.0 * ev


def make_synthetic_csv(
    path: Path,
    start: str = "2025-01-01",
    days: int = 365,
    freq: str = "1h",
    data: CampusData | None = None,
    seed: int = 0,
) -> Path:
    """Write a wide CSV following the README contract and return its path."""
    d = data or load_campus_data()
    rng = np.random.default_rng(seed)
    idx = pd.date_range(
        start, periods=int(days * pd.Timedelta("1D") / pd.Timedelta(freq)), freq=freq
    )
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    doy = idx.dayofyear.to_numpy()
    weekday = (idx.dayofweek.to_numpy() < 5).astype(float)

    winter = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - 15) / 365)  # 1 in Jan, 0 in Jul
    work = np.exp(-0.5 * ((hour - 13) / 4.5) ** 2) * (0.5 + 0.5 * weekday)
    sun = (
        np.clip(np.sin(np.pi * (hour - 6) / 13), 0, None)
        * (0.35 + 0.65 * (1 - winter))
        * (0.6 + 0.4 * rng.random(len(idx)))
    )

    cols = {"timestamp": idx}
    for r in d.loads.itertuples():
        kw = r.rated_kVA * r.power_factor
        cols[r.csv_column] = (
            kw
            * (0.18 + 0.15 * work + 0.05 * winter)
            * (1 + 0.05 * rng.standard_normal(len(idx)))
        )
    for r in d.heat_pumps.itertuples():
        cols[r.csv_column] = (
            r.rated_kVA * r.power_factor * (0.1 + 0.5 * winter) * (0.7 + 0.3 * work)
        )
    if d.has_heat_side:  # outdoor temperature: seasonal + diurnal swing, around 11 C
        temp = pd.Series(
            11
            - 12 * np.cos(2 * np.pi * (doy - 15) / 365)
            - 4 * np.cos(2 * np.pi * (hour - 3) / 24),
            index=idx,
        )
        cols.update(
            {k: v.to_numpy() for k, v in heat_columns(temp, d).items()}
        )
    cols["pv_per_kwp"] = 0.85 * sun
    base_ev = ev_per_charger_profile(idx)  # kW per charger
    for k, c in enumerate(d.ev_chargers.itertuples(), start=1):
        cols[c.csv_column] = base_ev * (0.8 + 0.04 * k)
    df = pd.DataFrame(cols).set_index("timestamp").clip(lower=0).reset_index()
    # drop duplicates in columns shared by several tables (pv / ev site profiles)
    df = df.loc[:, ~df.columns.duplicated()]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return path
