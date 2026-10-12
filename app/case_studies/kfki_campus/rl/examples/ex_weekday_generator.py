"""Example 4 - a full generator subclass: when a sampler is not enough.

``JointDistributionGenerator`` draws the same number of arrivals every day and its sampler does
not know the day or the charger. Subclass ``ChargingEventGenerator`` when you need that: here
weekday and weekend have their own sampler and arrival rate, and each charger has its own rate.
"""

from typing import Optional

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.rl.examples._guards import SafeSampler
from app.case_studies.kfki_campus.rl.examples.ex_scipy_marginals import make_sampler
from app.operation.rl.events import (
    GENERATOR_REGISTRY,
    ChargingEvent,
    ChargingEventGenerator,
    drop_overlaps,
)


# [start:generator]
class WeekdayGenerator(ChargingEventGenerator):
    """Weekday / weekend samplers and a plug-in rate per charger.

    :ivar weekday: Sampler used Monday-Friday
    :ivar weekend: Sampler used Saturday-Sunday
    :ivar dict rates: Mean plug-ins per day for each charger id (``default_rate`` otherwise)
    :ivar float weekend_factor: Weekend rate relative to the weekday rate
    """

    def __init__(
        self,
        weekday,
        weekend,
        rates: Optional[dict] = None,
        default_rate: float = 1.5,
        weekend_factor: float = 0.4,
    ):
        self.weekday, self.weekend = weekday, weekend
        self.rates = rates or {}
        self.default_rate = default_rate
        self.weekend_factor = weekend_factor

    def sample(self, rng, start, end, chargers):
        # start one day early: vehicles connected at ``start`` came the day before
        days = pd.date_range(start.floor("D") - pd.Timedelta(days=1), end, freq="D")
        events = []
        for cid in chargers:
            for day in days:
                weekend = day.dayofweek >= 5
                rate = self.rates.get(cid, self.default_rate)
                n = rng.poisson(rate * (self.weekend_factor if weekend else 1.0))
                if n == 0:
                    continue
                hour, energy, dwell = (
                    self.weekend if weekend else self.weekday
                ).sample(n, rng)
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
        events = drop_overlaps(events)  # one vehicle per charger at a time
        return [e for e in events if e.plug_out > start and e.plug_in < end]


# [end:generator]


# [start:register]
GENERATOR_REGISTRY["weekday"] = (
    WeekdayGenerator  # now usable as ``events.generator: weekday``
)
# [end:register]


def make_generator(rates: Optional[dict] = None) -> WeekdayGenerator:
    """Both day types use the marginals of example 1 (replace by your own samplers)."""
    return WeekdayGenerator(
        SafeSampler(make_sampler()),
        SafeSampler(make_sampler()),
        rates=rates,
    )
