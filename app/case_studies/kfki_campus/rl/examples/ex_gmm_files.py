"""Example 5 - your own ``.gmm`` files: one Gaussian mixture for workdays, one for weekends.

A ``.gmm`` file is a joblib (pickle) dump of a fitted scikit-learn ``GaussianMixture`` over
three raw dimensions, in this order: start hour of the day, energy (kWh), connected time (h).
The same two mixtures are used for every charger. The number of arrivals is not part of a
mixture: it comes from ``workday_rate`` / ``weekend_rate`` (see ``arrival_rates``).

Needs scikit-learn (``pip install -r requirements-rl.txt``), in the version that saved the files.
"""

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.rl.examples._guards import SafeSampler
from app.case_studies.kfki_campus.rl.examples.ex_weekday_generator import (
    WeekdayGenerator,
)
from app.operation.rl.events import GENERATOR_REGISTRY


# [start:load]
def load_gmm(path):
    """Load a ``.gmm`` file and check that it is a mixture over (start hour, energy, time).

    Only open files you trust: pickle can execute code.
    """
    import joblib  # installed with scikit-learn

    gmm = joblib.load(path)
    means = getattr(gmm, "means_", None)
    if means is None or means.shape[1] != 3:
        raise ValueError(
            f"{path}: expected a fitted GaussianMixture over 3 dimensions "
            f"(start hour, energy kWh, connected h), got {type(gmm).__name__}"
            + ("" if means is None else f" with {means.shape[1]} dimensions")
        )
    return gmm


# [end:load]


# [start:sampler]
class GmmSampler:
    """Draw (start hour, energy kWh, connected h) from a fitted ``GaussianMixture``.

    The mixture is over the raw values, so it can return negative energies or times and hours
    outside 0..24: wrap it in ``SafeSampler`` (it redraws those rows).
    """

    def __init__(self, gmm):
        self.gmm = gmm

    def sample(self, n, rng):
        # GaussianMixture.sample cannot take our rng: reseed it from it at every call,
        # otherwise every call (every day) would return the same numbers.
        self.gmm.random_state = int(rng.integers(2**31))
        x, _ = self.gmm.sample(n)
        return x[:, 0], x[:, 1], x[:, 2]


# [end:sampler]


# [start:rates]
def arrival_rates(sessions: pd.DataFrame) -> tuple[float, float]:
    """Mean plug-ins per charger and day on workdays and on weekends, from a session table.

    :param pd.DataFrame sessions: Columns ``charger_id`` and ``plug_in`` (the session file)
    :returns: ``(workday_rate, weekend_rate)``
    """
    t = pd.to_datetime(sessions["plug_in"])
    days = pd.date_range(t.min().normalize(), t.max().normalize(), freq="D")
    per_day = t.dt.normalize().value_counts().reindex(days, fill_value=0)
    n_chargers = sessions["charger_id"].nunique()
    workday = per_day[days.dayofweek < 5].mean() / n_chargers
    weekend = per_day[days.dayofweek >= 5].mean() / n_chargers
    return float(workday), float(weekend)


# [end:rates]


# [start:generator]
class GmmFilesGenerator(WeekdayGenerator):
    """Events from a workday and a weekend ``.gmm`` file (the same for all chargers).

    :param workday: Path of the workday ``.gmm`` file
    :param weekend: Path of the weekend ``.gmm`` file
    :param float workday_rate: Mean plug-ins per charger on a workday
    :param float weekend_rate: Mean plug-ins per charger on a weekend day
    """

    def __init__(self, workday, weekend, workday_rate=1.5, weekend_rate=0.6):
        super().__init__(
            SafeSampler(GmmSampler(load_gmm(workday))),
            SafeSampler(GmmSampler(load_gmm(weekend))),
            default_rate=workday_rate,
            weekend_factor=weekend_rate / workday_rate,
        )


GENERATOR_REGISTRY["gmm_files"] = GmmFilesGenerator  # usable as ``events.generator``
# [end:generator]


def make_generator(
    workday, weekend, workday_rate: float = 1.5, weekend_rate: float = 0.6
) -> GmmFilesGenerator:
    """Generator from the two files (see the guide, step 4.4)."""
    return GmmFilesGenerator(workday, weekend, workday_rate, weekend_rate)


def fit_demo_gmm(path, weekend: bool = False, seed: int = 0, n: int = 800):
    """Fit and save a made-up ``.gmm`` (to try the example without your files).

    Raw dimensions (start hour, energy kWh, connected h); NOT measured data.
    """
    import joblib
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(seed)
    hour = rng.normal(11.0 if weekend else 8.5, 3.0 if weekend else 1.5, n)
    energy = np.clip(rng.lognormal(np.log(12.0), 0.5, n), 2.0, 60.0)
    dwell = energy / 7.0 * 1.4 + rng.gamma(2.0, 1.0, n)
    gmm = GaussianMixture(3, random_state=seed).fit(
        np.column_stack([hour, energy, dwell])
    )
    joblib.dump(gmm, path)
    return path
