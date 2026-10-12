"""Safety net and checks for samplers of charging events (used by the examples).

``JointDistributionGenerator`` does not validate what a sampler returns. A kernel-density
estimate or a normal distribution can return negative energies or hours outside 0..24, which
would silently produce nonsense sessions. ``SafeSampler`` wraps any sampler and fixes that;
``check_sampler`` / ``plot_sampler`` show what a sampler produces before it is used.

A sampler is any object with ``sample(n, rng) -> (start_hour, energy_kwh, connected_h)``:
three arrays of length ``n``, start as hour of day, energy in kWh, connected time in hours.
"""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ("start_hour", "energy_kwh", "connected_h")


class SafeSampler:
    """Validate and repair the output of another sampler.

    * the three arrays must have length ``n`` (otherwise ``ValueError``);
    * rows with non-finite or non-positive energy / connected time are drawn again;
    * the start hour is wrapped into ``[0, 24)``.

    :ivar sampler: Wrapped sampler
    :ivar int max_tries: Redraw rounds before giving up (``RuntimeError``)
    """

    def __init__(self, sampler, max_tries: int = 20):
        self.sampler = sampler
        self.max_tries = max_tries

    def sample(self, n: int, rng: np.random.Generator):
        have = np.empty((0, 3))
        for _ in range(self.max_tries):
            need = n - len(have)
            if need <= 0:
                break
            draw = self.sampler.sample(need, rng)
            cols = [np.asarray(c, dtype=float).reshape(-1) for c in draw]
            if len(cols) != 3 or any(len(c) != need for c in cols):
                raise ValueError(
                    f"sampler must return 3 arrays of length {need} "
                    f"(start_hour, energy_kwh, connected_h), got {[len(c) for c in cols]}"
                )
            rows = np.column_stack(cols)
            ok = np.isfinite(rows).all(axis=1) & (rows[:, 1] > 0) & (rows[:, 2] > 0)
            have = np.vstack([have, rows[ok]])
        if len(have) < n:
            raise RuntimeError(
                f"sampler returned too many invalid rows (energy or time <= 0): "
                f"{len(have)} of {n} valid after {self.max_tries} tries"
            )
        have = have[:n]
        return have[:, 0] % 24.0, have[:, 1], have[:, 2]


def check_sampler(sampler, n: int = 10_000, seed: int = 0):
    """Draw ``n`` samples and summarise them.

    :returns: ``(summary, correlation)`` DataFrames. Look for: start hours in [0, 24),
        energy and connected time positive and in a plausible range (kWh, hours - not Wh or
        minutes), and the correlations you expect (long stays go with large energies?).
    """
    rng = np.random.default_rng(seed)
    cols = sampler.sample(n, rng)
    df = pd.DataFrame({c: np.asarray(v, dtype=float) for c, v in zip(COLUMNS, cols)})
    summary = df.describe(percentiles=[0.05, 0.5, 0.95]).T
    summary["share_non_positive"] = (df <= 0).mean()
    return summary, df.corr()


def plot_sampler(sampler, path: Path, n: int = 10_000, seed: int = 0) -> Path:
    """Histograms of the three variables and the energy-vs-connected-time scatter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    hour, energy, dwell = (np.asarray(v, dtype=float) for v in sampler.sample(n, rng))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    axes[0].hist(hour, bins=np.arange(0, 25), color="#2a78d6")
    axes[0].set_xlabel("start hour of day")
    axes[1].hist(energy, bins=40, color="#2a78d6")
    axes[1].set_xlabel("energy (kWh)")
    axes[2].hist(dwell, bins=40, color="#2a78d6")
    axes[2].set_xlabel("connected time (h)")
    axes[3].scatter(dwell[:2000], energy[:2000], s=4, alpha=0.4, color="#2a78d6")
    axes[3].set_xlabel("connected time (h)")
    axes[3].set_ylabel("energy (kWh)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def demo_sessions(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Made-up sessions (NOT measured data) to run the examples without a data file.

    Columns follow the session file: ``plug_in``, ``connected_h``, ``energy_kwh``. Arrivals
    cluster in the morning (workplace) and in the evening (residential); larger energies
    stay longer.
    """
    rng = np.random.default_rng(seed)
    morning = rng.random(n) < 0.6
    hour = np.where(morning, rng.normal(8.5, 1.2, n), rng.normal(18.0, 1.8, n)) % 24
    energy = np.clip(rng.lognormal(np.log(12.0), 0.6, n), 1.0, 70.0)
    dwell = energy / 7.0 * rng.uniform(1.0, 1.8, n) + rng.gamma(2.0, 1.0, n)
    day = pd.to_datetime("2025-06-02") + pd.to_timedelta(rng.integers(0, 60, n), "D")
    return pd.DataFrame(
        {
            "plug_in": day + pd.to_timedelta(hour, "h"),
            "connected_h": dwell,
            "energy_kwh": energy,
        }
    )
