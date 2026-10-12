"""Example 3 - bootstrap: draw whole sessions from an empirical table.

Use it when you have a table of measured sessions (e.g. your session file) and want random
days built from real sessions. Nothing is assumed about the shape of the distribution.
"""

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.rl.examples._guards import SafeSampler, demo_sessions
from app.operation.rl.events import JointDistributionGenerator


# [start:sampler]
class BootstrapSampler:
    """Resample rows of a sessions table, with a little noise so that days differ.

    :param pd.DataFrame sessions: Columns ``plug_in``, ``connected_h``, ``energy_kwh``
    :param float jitter: Relative noise of energy and connected time (0 = exact copies)
    :param float hour_jitter: Noise of the start hour in hours
    """

    def __init__(
        self, sessions: pd.DataFrame, jitter: float = 0.1, hour_jitter: float = 0.25
    ):
        t = pd.to_datetime(sessions["plug_in"])
        self.hour = (t.dt.hour + t.dt.minute / 60.0).to_numpy()
        self.energy = sessions["energy_kwh"].to_numpy(dtype=float)
        self.dwell = sessions["connected_h"].to_numpy(dtype=float)
        self.jitter, self.hour_jitter = jitter, hour_jitter

    def sample(self, n, rng):
        i = rng.integers(0, len(self.hour), size=n)  # whole rows keep the correlations
        noise = lambda: rng.lognormal(0.0, self.jitter, size=n)  # noqa: E731
        hour = self.hour[i] + rng.normal(0.0, self.hour_jitter, size=n)
        return hour, self.energy[i] * noise(), self.dwell[i] * noise()


# [end:sampler]


def make_generator(
    sessions: pd.DataFrame | None = None, arrivals_per_day: float = 1.5
) -> JointDistributionGenerator:
    """Generator from the given sessions (default: made-up demo sessions).

    With a session file: ``sessions = pd.read_csv("my_sessions.csv")``.
    """
    sessions = demo_sessions() if sessions is None else sessions
    return JointDistributionGenerator(
        SafeSampler(BootstrapSampler(sessions)), arrivals_per_day
    )
