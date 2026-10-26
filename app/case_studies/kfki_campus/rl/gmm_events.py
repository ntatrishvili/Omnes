"""Charging events from the measured Gaussian mixtures of the campus (``*.gmm`` bundles).

A bundle is a joblib dict ``{"gmm": GaussianMixture, "scaler": StandardScaler, "features":
["shifted_start_time_sec", "energy_kwh", "connected_h"], "time_shift_hour": 4, ...}``: the
mixture lives in standardised space, the start time is shifted by ``time_shift_hour`` so that
the arrivals do not wrap around midnight (``start = shifted + time_shift_hour``). One bundle
describes working days, one weekends / holidays; the same pair is used for every charger.

The mixtures give start, energy and stay; the number of plug-ins is not part of them:
``workday_rate`` / ``weekend_rate`` are the mean plug-in attempts per charger and day. A
charger holds one vehicle at a time, so an attempt that falls into a stay in progress is
dropped (``drop_overlaps``): the sessions actually served per charger are fewer than the rate.

Clock: the bundles say ``model_clock: Etc/GMT-1``, the fixed UTC+1 clock of the model (the
Budapest wall-clock data were converted when the mixtures were fitted); ``clock="local"``
treats the drawn start as Budapest wall-clock time (CET / CEST) and converts it instead.

Only open files you trust: joblib / pickle can execute code.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.operation.rl.events import (
    GENERATOR_REGISTRY,
    ChargingEvent,
    ChargingEventGenerator,
    drop_overlaps,
)

INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "input"
WORKDAY_FILE = INPUT_DIR / "weekday_workplace_rl3d.gmm"
WEEKEND_FILE = INPUT_DIR / "weekend_workplace_rl3d.gmm"
FEATURES = ["shifted_start_time_sec", "energy_kwh", "connected_h"]


def load_bundle(path) -> dict:
    """Load a ``.gmm`` bundle and check that it has the expected parts and features."""
    import joblib  # installed with scikit-learn

    b = joblib.load(path)
    missing = [k for k in ("gmm", "scaler", "features") if not isinstance(b, dict) or k not in b]
    if missing:
        raise ValueError(f"{path}: not a gmm bundle (missing {missing or 'dict'})")
    if list(b["features"]) != FEATURES:
        raise ValueError(f"{path}: features {b['features']}, expected {FEATURES}")
    return b


class BundleSampler:
    """(start hour of day, energy kWh, connected h) from a bundle; rows are drawn again until valid.

    The mixture is Gaussian, so it can give negative or absurd values: rows outside
    ``[min_energy_kwh, max_energy_kwh]`` / ``[min_stay_h, max_stay_h]`` are redrawn.
    """

    def __init__(
        self,
        bundle: dict,
        min_energy_kwh: float = 1.0,
        max_energy_kwh: float = 80.0,
        min_stay_h: float = 0.25,
        max_stay_h: float = 48.0,
    ):
        self.gmm, self.scaler = bundle["gmm"], bundle["scaler"]
        self.shift_h = float(bundle.get("time_shift_hour", 0))
        self.lim = (min_energy_kwh, max_energy_kwh, min_stay_h, max_stay_h)

    def sample(self, n: int, rng: np.random.Generator):
        e0, e1, d0, d1 = self.lim
        have = np.empty((0, 3))
        for _ in range(50):
            need = n - len(have)
            if need <= 0:
                break
            # GaussianMixture.sample cannot take our rng: reseed it from it at every call
            self.gmm.random_state = int(rng.integers(2**31))
            x, _ = self.gmm.sample(max(2 * need, 8))
            x = self.scaler.inverse_transform(x)
            ok = (x[:, 1] >= e0) & (x[:, 1] <= e1) & (x[:, 2] >= d0) & (x[:, 2] <= d1)
            have = np.vstack([have, x[ok]])
        if len(have) < n:
            raise RuntimeError("the mixture returns too many invalid sessions")
        have = have[:n]
        start = (have[:, 0] / 3600.0 + self.shift_h) % 24.0
        return start, have[:, 1], have[:, 2]


def _is_workday(day: pd.Timestamp, holidays_hu) -> bool:
    return day.dayofweek < 5 and (holidays_hu is None or day.date() not in holidays_hu)


class GmmBundleGenerator(ChargingEventGenerator):
    """Events from a working-day and a weekend ``.gmm`` bundle (the same for all chargers).

    :param workday: Bundle for working days (default: the campus file in ``data/input``)
    :param weekend: Bundle for weekends and holidays
    :param float workday_rate: Mean plug-in attempts per charger on a working day
    :param float weekend_rate: Mean plug-in attempts per charger on a weekend / holiday
    :param str clock: ``model`` (the drawn start is on the model clock, as the bundles say) |
        ``local`` (Budapest wall-clock time, converted to the fixed UTC+1 model clock)
    :param float stay_scale: Multiplies every stay (< 1: shorter stays, less flexibility)
    :param float energy_scale: Multiplies every requested energy
    :param bool holidays: Treat Hungarian public holidays as weekend days (needs the optional
        ``holidays`` package; ignored with a warning-free fallback to weekdays if it is missing)
    """

    def __init__(
        self,
        workday: Optional[str] = None,
        weekend: Optional[str] = None,
        workday_rate: float = 4.0,
        weekend_rate: float = 1.6,
        clock: str = "model",
        stay_scale: float = 1.0,
        energy_scale: float = 1.0,
        holidays: bool = True,
    ):
        if clock not in ("model", "local"):
            raise ValueError("clock must be 'model' or 'local'")
        self.workday = BundleSampler(load_bundle(workday or WORKDAY_FILE))
        self.weekend = BundleSampler(load_bundle(weekend or WEEKEND_FILE))
        self.workday_rate, self.weekend_rate = workday_rate, weekend_rate
        self.clock, self.stay_scale, self.energy_scale = clock, stay_scale, energy_scale
        self._holidays = None
        if holidays:
            try:
                import holidays as _h

                self._holidays = _h.country_holidays("HU")
            except ImportError:
                pass

    def _plug_in(self, day: pd.Timestamp, hour: float) -> pd.Timestamp:
        t = day + pd.Timedelta(hours=float(hour))
        if self.clock == "model":
            return t
        wall = t.tz_localize("Europe/Budapest", ambiguous=True, nonexistent="shift_forward")
        return wall.tz_convert("Etc/GMT-1").tz_localize(None)

    def sample(self, rng, start, end, chargers):
        # start one day early: vehicles connected at ``start`` came the day before
        days = pd.date_range(start.floor("D") - pd.Timedelta(days=1), end, freq="D")
        slots = []  # (charger, day, working day, plug-in attempts)
        for cid in chargers:
            for day in days:
                work = _is_workday(day, self._holidays)
                n = int(rng.poisson(self.workday_rate if work else self.weekend_rate))
                if n:
                    slots.append((cid, day, work, n))
        # one draw per day type for the whole window: the mixture sampler has a large
        # per-call overhead and an episode reset would otherwise call it dozens of times
        draws = {}
        for work, sampler in ((True, self.workday), (False, self.weekend)):
            total = sum(n for _, _, w, n in slots if w == work)
            draws[work] = [iter(c) for c in sampler.sample(total, rng)] if total else None
        events = []
        for cid, day, work, n in slots:
            for k in range(n):
                hour, energy, stay = (next(c) for c in draws[work])
                events.append(
                    ChargingEvent(
                        f"{cid}_{day:%Y%m%d}_{k}",
                        cid,
                        self._plug_in(day, hour),
                        float(stay * self.stay_scale),
                        float(energy * self.energy_scale),
                    )
                )
        events = drop_overlaps(events)  # one vehicle per charger at a time
        return [e for e in events if e.plug_out > start and e.plug_in < end]


GENERATOR_REGISTRY["gmm_bundle"] = GmmBundleGenerator  # ``events.generator: gmm_bundle``
