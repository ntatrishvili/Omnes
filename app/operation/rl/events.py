"""Charging events and their generators.

A ``ChargingEvent`` is one plug-in: when the vehicle connects, how long it stays and how much
energy it must receive. The environment only talks to ``ChargingEventGenerator.sample``, so
deterministic events (from a session file) and random events (any sampler) are interchangeable.
Specialise by subclassing ``ChargingEventGenerator`` and adding it to ``GENERATOR_REGISTRY``;
the name goes into ``EventSpec.generator``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChargingEvent:
    """One charging session.

    :ivar str event_id: Unique id
    :ivar str charger_id: Charger the vehicle plugs into
    :ivar pd.Timestamp plug_in: Connection time (model clock)
    :ivar float connected_h: Time the vehicle stays connected (h)
    :ivar float energy_kwh: Energy to deliver (AC side)
    :ivar float max_power_kw: Vehicle / charger limit; ``None`` = charger rating
    """

    event_id: str
    charger_id: str
    plug_in: pd.Timestamp
    connected_h: float
    energy_kwh: float
    max_power_kw: Optional[float] = None

    @property
    def plug_out(self) -> pd.Timestamp:
        return self.plug_in + pd.Timedelta(hours=self.connected_h)


class ChargingEventGenerator(ABC):
    """Source of charging events."""

    @abstractmethod
    def sample(
        self,
        rng: np.random.Generator,
        start: pd.Timestamp,
        end: pd.Timestamp,
        chargers: dict[str, float],
    ) -> list[ChargingEvent]:
        """Events overlapping ``[start, end)``.

        :param rng: Random generator (deterministic generators ignore it)
        :param chargers: Rated power (kW) per charger id; events only for these chargers
        :returns: Events, non-overlapping per charger
        """


def drop_overlaps(events: Sequence[ChargingEvent]) -> list[ChargingEvent]:
    """Keep the earlier of two events of one charger that overlap (rejection)."""
    kept: list[ChargingEvent] = []
    free_at: dict[str, pd.Timestamp] = {}
    for e in sorted(events, key=lambda e: (e.charger_id, e.plug_in)):
        if e.plug_in >= free_at.get(e.charger_id, pd.Timestamp.min):
            kept.append(e)
            free_at[e.charger_id] = e.plug_out
    return sorted(kept, key=lambda e: e.plug_in)


class SessionFileGenerator(ChargingEventGenerator):
    """Deterministic events (e.g. read from the user's session file)."""

    def __init__(self, events: Sequence[ChargingEvent]):
        self.events = list(events)

    def sample(self, rng, start, end, chargers):
        return [
            e
            for e in self.events
            if e.charger_id in chargers and e.plug_out > start and e.plug_in < end
        ]


# arrival probability per hour of day (unnormalised): morning and evening peaks
_WEEKDAY_HOURS = np.array(
    [1, 1, 1, 1, 1, 2, 4, 8, 12, 10, 6, 5, 5, 5, 5, 6, 8, 10, 8, 5, 3, 2, 1, 1], float
)
_WEEKEND_HOURS = np.array(
    [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 5, 5, 4, 3, 2, 2, 1, 1], float
)


class DummyChargingEventGenerator(ChargingEventGenerator):
    """Placeholder random events with **independent** marginals.

    Arrivals per charger are Poisson with an hour-of-day profile (weekday / weekend); the
    energy is lognormal; the connected time is the time needed at rated power times
    ``flex_factor`` plus a lognormal extra dwell. Start, energy and connected time are NOT
    correlated beyond that: replace it by a sampler of the measured joint distribution
    (``JointDistributionGenerator``) when available.

    :ivar float arrivals_per_day: Mean plug-ins per charger and weekday
    :ivar float weekend_factor: Arrival rate on weekends relative to weekdays
    :ivar float energy_median_kwh: Median requested energy
    :ivar float energy_sigma: Log-sigma of the energy
    :ivar float energy_max_kwh: Upper cut of the energy
    :ivar float flex_factor: Connected time / charging time at rated power (>= 1)
    :ivar float dwell_median_h: Median extra dwell time
    :ivar float dwell_sigma: Log-sigma of the extra dwell time
    """

    def __init__(
        self,
        arrivals_per_day: float = 1.5,
        weekend_factor: float = 0.4,
        energy_median_kwh: float = 14.0,
        energy_sigma: float = 0.6,
        energy_max_kwh: float = 80.0,
        flex_factor: float = 1.3,
        dwell_median_h: float = 2.0,
        dwell_sigma: float = 0.7,
        hour_weights_weekday: Optional[Sequence[float]] = None,
        hour_weights_weekend: Optional[Sequence[float]] = None,
    ):
        self.arrivals_per_day = arrivals_per_day
        self.weekend_factor = weekend_factor
        self.energy_median_kwh = energy_median_kwh
        self.energy_sigma = energy_sigma
        self.energy_max_kwh = energy_max_kwh
        self.flex_factor = flex_factor
        self.dwell_median_h = dwell_median_h
        self.dwell_sigma = dwell_sigma
        self.hours_wd = np.asarray(
            _WEEKDAY_HOURS if hour_weights_weekday is None else hour_weights_weekday,
            float,
        )
        self.hours_we = np.asarray(
            _WEEKEND_HOURS if hour_weights_weekend is None else hour_weights_weekend,
            float,
        )

    def _event(self, rng, cid, rated, day, weekend, n) -> ChargingEvent:
        w = self.hours_we if weekend else self.hours_wd
        hour = rng.choice(24, p=w / w.sum()) + rng.random()
        energy = float(
            np.clip(
                rng.lognormal(np.log(self.energy_median_kwh), self.energy_sigma),
                1.0,
                self.energy_max_kwh,
            )
        )
        dwell = energy / rated * self.flex_factor + rng.lognormal(
            np.log(self.dwell_median_h), self.dwell_sigma
        )
        return ChargingEvent(
            event_id=f"{cid}_{day:%Y%m%d}_{n}",
            charger_id=cid,
            plug_in=day + pd.Timedelta(hours=float(hour)),
            connected_h=float(dwell),
            energy_kwh=energy,
        )

    def sample(self, rng, start, end, chargers):
        # start one day early so that sessions already connected at ``start`` exist
        days = pd.date_range(start.floor("D") - pd.Timedelta(days=1), end, freq="D")
        events = []
        for cid, rated in chargers.items():
            for day in days:
                weekend = day.dayofweek >= 5
                rate = self.arrivals_per_day * (self.weekend_factor if weekend else 1.0)
                for n in range(rng.poisson(rate)):
                    events.append(self._event(rng, cid, rated, day, weekend, n))
        events = drop_overlaps(events)
        return [e for e in events if e.plug_out > start and e.plug_in < end]


class JointDistributionGenerator(ChargingEventGenerator):
    """Events drawn from a joint distribution of (start, energy, connected time).

    The sampler is any object with ``sample(n, rng) -> (start_hour, energy_kwh, connected_h)``
    (arrays of length ``n``; ``start_hour`` = hour of day in ``[0, 24)``), e.g. a KDE, a
    Gaussian mixture or a copula fitted to measured sessions. Not integrated yet: without a
    sampler it raises ``NotImplementedError``.

    :ivar float arrivals_per_day: Mean plug-ins per charger and day
    """

    def __init__(self, sampler=None, arrivals_per_day: float = 1.5):
        self.sampler = sampler
        self.arrivals_per_day = arrivals_per_day

    def sample(self, rng, start, end, chargers):
        if self.sampler is None:
            raise NotImplementedError(
                "JointDistributionGenerator needs a sampler with sample(n, rng) -> "
                "(start_hour, energy_kwh, connected_h)"
            )
        days = pd.date_range(start.floor("D") - pd.Timedelta(days=1), end, freq="D")
        events = []
        for cid in chargers:
            for day in days:
                n = rng.poisson(self.arrivals_per_day)
                if n == 0:
                    continue
                hour, energy, dwell = self.sampler.sample(n, rng)
                for k in range(n):
                    events.append(
                        ChargingEvent(
                            f"{cid}_{day:%Y%m%d}_{k}",
                            cid,
                            day + pd.Timedelta(hours=float(hour[k])),
                            float(dwell[k]),
                            float(energy[k]),
                        )
                    )
        events = drop_overlaps(events)
        return [e for e in events if e.plug_out > start and e.plug_in < end]


GENERATOR_REGISTRY: dict[str, type[ChargingEventGenerator]] = {
    "dummy": DummyChargingEventGenerator,
    "joint": JointDistributionGenerator,
}
