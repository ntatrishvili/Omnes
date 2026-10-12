"""Example 2 - a fitted joint distribution (kernel density estimate of measured sessions).

Use it when you have sessions and want to keep the correlations between start, energy and
connected time. The same pattern works with any fitted object that can draw samples.
"""

import numpy as np
import pandas as pd
from scipy import stats

from app.case_studies.kfki_campus.rl.examples._guards import SafeSampler, demo_sessions
from app.operation.rl.events import JointDistributionGenerator


# [start:sampler]
class KdeSampler:
    """Joint density of (start hour, log energy, log connected time).

    Logs keep energy and time positive after the transformation back; ``SafeSampler``
    wraps the start hour into [0, 24) and redraws anything that is still invalid.

    :param pd.DataFrame sessions: Columns ``plug_in``, ``connected_h``, ``energy_kwh``
    """

    def __init__(self, sessions: pd.DataFrame):
        t = pd.to_datetime(sessions["plug_in"])
        hour = t.dt.hour + t.dt.minute / 60.0
        data = np.vstack(
            [hour, np.log(sessions["energy_kwh"]), np.log(sessions["connected_h"])]
        )
        self.kde = stats.gaussian_kde(data)  # data shape: (variables, observations)

    def sample(self, n, rng):
        hour, log_energy, log_dwell = self.kde.resample(n, seed=rng)  # pass the rng
        return hour, np.exp(log_energy), np.exp(log_dwell)


# [end:sampler]


# A scikit-learn mixture works the same way (NOT tested here; scikit-learn is not a
# dependency of this project):
#
#     from sklearn.mixture import GaussianMixture
#     gm = GaussianMixture(3, random_state=0).fit(data.T)
#     ...
#     def sample(self, n, rng):
#         gm.random_state = int(rng.integers(2**31))   # reseed from OUR rng at every call
#         x, _ = gm.sample(n)
#         return x[:, 0], np.exp(x[:, 1]), np.exp(x[:, 2])
#
# ``GaussianMixture.sample`` cannot take our ``rng``. A fixed ``random_state`` would return
# the same numbers at every call (every day identical), so reseed it from the rng each time.


def make_generator(
    sessions: pd.DataFrame | None = None, arrivals_per_day: float = 1.5
) -> JointDistributionGenerator:
    """Generator from the given sessions (default: made-up demo sessions)."""
    sessions = demo_sessions() if sessions is None else sessions
    return JointDistributionGenerator(
        SafeSampler(KdeSampler(sessions)), arrivals_per_day
    )
