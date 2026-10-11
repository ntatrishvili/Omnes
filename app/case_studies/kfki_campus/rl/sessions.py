"""Charging-session input file: read, validate, write, and turn into an uncontrolled profile.

File format (CSV, one row per plug-in, timestamps naive on the fixed UTC+1 clock of the
time-series CSV, ``.`` decimal):

=============== ============ ========================================================
column          required     meaning
=============== ============ ========================================================
session_id      yes          unique id
charger_id      yes          id of ``assets.xlsx:ev_chargers`` (``ev_c01`` ...)
plug_in         yes          connection time (ISO 8601)
connected_h     yes or       time the vehicle stays connected (h)
plug_out        alternative  disconnection time; both given -> must agree (1 min)
energy_kwh      yes          energy to deliver at the charger (AC side), > 0
max_power_kw    no           vehicle limit; default = charger rating
=============== ============ ========================================================
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.infra.logging_setup import get_logger
from app.operation.rl.events import ChargingEvent

log = get_logger(__name__)

REQUIRED = ["session_id", "charger_id", "plug_in", "energy_kwh"]
COLUMNS = [
    "session_id",
    "charger_id",
    "plug_in",
    "connected_h",
    "plug_out",
    "energy_kwh",
    "max_power_kw",
]


def read_sessions(
    path: Path,
    chargers: dict[str, float],
    period: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> tuple[list[ChargingEvent], pd.DataFrame]:
    """Read and validate a session file.

    Infeasible sessions (``energy_kwh`` above rated power x connected time) are clipped to
    what can be delivered and listed in the returned quality report; every other problem
    raises ``ValueError``.

    :param Path path: Session CSV
    :param dict chargers: Rated power (kW) per known charger id
    :param period: Optional ``(start, end)``; sessions entirely outside are dropped
    :returns: ``(events, report)`` with one report row per clipped / dropped session
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{Path(path).name}: missing columns {missing}")
    if "connected_h" not in df.columns and "plug_out" not in df.columns:
        raise ValueError(f"{Path(path).name}: need 'connected_h' or 'plug_out'")
    if df["session_id"].duplicated().any():
        raise ValueError(f"{Path(path).name}: duplicate session_id")
    unknown = sorted(set(df["charger_id"]) - set(chargers))
    if unknown:
        raise ValueError(f"{Path(path).name}: unknown charger_id {unknown}")
    df["plug_in"] = pd.to_datetime(df["plug_in"])
    if "plug_out" in df.columns and df["plug_out"].notna().any():
        out = pd.to_datetime(df["plug_out"])
        derived = (out - df["plug_in"]).dt.total_seconds() / 3600.0
        if "connected_h" in df.columns:
            bad = df["connected_h"].notna() & out.notna()
            if ((df.loc[bad, "connected_h"] - derived[bad]).abs() > 1 / 60).any():
                raise ValueError("connected_h and plug_out disagree")
            df["connected_h"] = df["connected_h"].fillna(derived)
        else:
            df["connected_h"] = derived
    if (df["connected_h"] <= 0).any() or df["connected_h"].isna().any():
        raise ValueError("connected_h must be positive (plug_out after plug_in)")
    if (df["energy_kwh"] <= 0).any():
        raise ValueError("energy_kwh must be positive")
    if "max_power_kw" not in df.columns:
        df["max_power_kw"] = np.nan

    report, events = [], []
    for r in df.sort_values("plug_in").itertuples():
        rated = chargers[r.charger_id]
        pmax = rated if pd.isna(r.max_power_kw) else min(rated, float(r.max_power_kw))
        energy = float(r.energy_kwh)
        if period is not None and not (
            r.plug_in + pd.Timedelta(hours=r.connected_h) > period[0]
            and r.plug_in < period[1]
        ):
            report.append((r.session_id, "outside period", energy, 0.0))
            continue
        if energy > pmax * r.connected_h + 1e-9:
            report.append(
                (r.session_id, "clipped to power x time", energy, pmax * r.connected_h)
            )
            energy = pmax * r.connected_h
        events.append(
            ChargingEvent(
                str(r.session_id),
                r.charger_id,
                r.plug_in,
                float(r.connected_h),
                energy,
                None if pd.isna(r.max_power_kw) else float(r.max_power_kw),
            )
        )
    for cid in {e.charger_id for e in events}:
        ev = sorted((e for e in events if e.charger_id == cid), key=lambda e: e.plug_in)
        for a, b in zip(ev, ev[1:]):
            if b.plug_in < a.plug_out - pd.Timedelta(seconds=1):
                raise ValueError(
                    f"overlapping sessions {a.event_id} / {b.event_id} on {cid}"
                )
    rep = pd.DataFrame(
        report, columns=["session_id", "issue", "energy_kwh", "energy_kept_kwh"]
    )
    if len(rep):
        log.warning(f"{len(rep)} session(s) adjusted or dropped in {Path(path).name}")
    return events, rep


def write_sessions(events: list[ChargingEvent], path: Path) -> Path:
    """Write events in the session-file format."""
    rows = [
        {
            "session_id": e.event_id,
            "charger_id": e.charger_id,
            "plug_in": e.plug_in.strftime("%Y-%m-%d %H:%M:%S"),
            "connected_h": round(e.connected_h, 4),
            "energy_kwh": round(e.energy_kwh, 3),
            "max_power_kw": e.max_power_kw,
        }
        for e in sorted(events, key=lambda e: e.plug_in)
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return Path(path)


def sessions_to_profile(
    events: list[ChargingEvent],
    index: pd.DatetimeIndex,
    chargers: dict[str, float],
) -> pd.DataFrame:
    """Uncontrolled demand (kW per charger): full power from plug-in until the energy is delivered.

    The result has one column per charger id on ``index`` (regular grid) and can be written
    to the ``ev_charger_NN`` columns of the time-series CSV.
    """
    dt_h = pd.Timedelta(index.freq or (index[1] - index[0])) / pd.Timedelta("1h")
    out = pd.DataFrame(0.0, index=index, columns=list(chargers))
    edges = index.append(pd.DatetimeIndex([index[-1] + pd.Timedelta(hours=dt_h)]))
    for e in events:
        p = min(chargers[e.charger_id], e.max_power_kw or np.inf)
        t_end = min(e.plug_out, e.plug_in + pd.Timedelta(hours=e.energy_kwh / p))
        # overlap (h) of [plug_in, t_end) with every grid step, as a mean power over the step
        lo = np.maximum(edges[:-1].values, np.datetime64(e.plug_in))
        hi = np.minimum(edges[1:].values, np.datetime64(t_end))
        overlap_h = np.clip((hi - lo) / np.timedelta64(1, "h"), 0.0, dt_h)
        out[e.charger_id] += p * overlap_h / dt_h
    return out
