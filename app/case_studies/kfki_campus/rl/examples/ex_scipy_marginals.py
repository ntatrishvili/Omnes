"""Example 1 - independent marginals from ``scipy.stats`` distributions.

Use it when you have one fitted distribution per variable (start hour profile, energy,
connected time) but no joint model. Start, energy and connected time are then independent.
"""

import numpy as np
from scipy import stats

from app.case_studies.kfki_campus.rl.examples._guards import SafeSampler
from app.operation.rl.events import JointDistributionGenerator


# [start:sampler]
class MarginalsSampler:
    """Independent start hour, energy and connected time.

    :ivar np.ndarray hour_weights: 24 weights, arrival probability per hour of the day
    :ivar energy: Frozen scipy distribution of the energy (kWh)
    :ivar dwell: Frozen scipy distribution of the connected time (h)
    """

    def __init__(self, hour_weights, energy, dwell):
        w = np.asarray(hour_weights, dtype=float)
        self.hour_p = w / w.sum()
        self.energy = energy
        self.dwell = dwell

    def sample(self, n, rng):
        hour = rng.choice(24, size=n, p=self.hour_p) + rng.random(n)  # hour + minutes
        energy = self.energy.rvs(size=n, random_state=rng)  # always pass the given rng
        dwell = self.dwell.rvs(size=n, random_state=rng)
        return hour, energy, dwell


# [end:sampler]


def make_sampler() -> MarginalsSampler:
    """Made-up parameters: replace them by your fitted ones."""
    hours = np.array(
        [1, 1, 1, 1, 1, 2, 4, 8, 12, 10, 6, 5, 5, 5, 5, 6, 8, 10, 8, 5, 3, 2, 1, 1]
    )
    energy = stats.lognorm(s=0.6, scale=12.0)  # median 12 kWh
    dwell = stats.gamma(a=3.0, scale=1.5)  # mean 4.5 h
    return MarginalsSampler(hours, energy, dwell)


def make_generator(arrivals_per_day: float = 1.5) -> JointDistributionGenerator:
    """Generator with the sampler wrapped in the safety net."""
    return JointDistributionGenerator(SafeSampler(make_sampler()), arrivals_per_day)
