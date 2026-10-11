"""Figures of the electricity-heat coupling (heat pumps + hot-water tanks).

Style as in plots_timeseries.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.case_studies.kfki_campus.plots import BLUE, INK2, MUTED, ORANGE  # noqa: E402
from app.case_studies.kfki_campus.plots_timeseries import (  # noqa: E402
    AQUA,
    YELLOW,
    _load_run,
    _save,
)


def heat_week(
    details: Path, sid: str, start: str, out: Path, days: int = 7
) -> Path | None:
    """One week of a heat-coupled scenario: electric side, heat side and tank state.

    The heat pump electricity (top panel) is placed by the optimiser into hours with PV
    surplus; the tank (bottom panel) decouples it from the heat demand (middle panel).
    """
    if not (Path(details) / f"{sid}.npz").exists():
        return None
    d = _load_run(details, sid)
    if "p_hp" not in d:
        return None
    t = d["time"]
    m = (t >= pd.Timestamp(start)) & (t < pd.Timestamp(start) + pd.Timedelta(days=days))
    if not m.any():
        return None
    tt = t[m]
    fig, ax = plt.subplots(
        3,
        1,
        figsize=(10, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.0, 1.2]},
    )
    base = (d["demand"] - d["p_hp"])[m] / 1000
    ax[0].fill_between(tt, base, color=MUTED, alpha=0.35, lw=0, label="other demand")
    ax[0].fill_between(
        tt, base, base + d["p_hp"][m] / 1000, color=ORANGE, alpha=0.9, lw=0,
        label="heat pumps",
    )
    ax[0].plot(tt, d["pv"][m] / 1000, color=YELLOW, lw=1.4, label="PV")
    ax[0].plot(
        tt,
        (d["p_import"][m] - d["p_export"][m]) / 1000,
        color=BLUE,
        lw=1.2,
        label="grid (+import / -export)",
    )
    ax[0].set_ylabel("electric MW")
    ax[0].legend(fontsize=8, ncol=4, loc="upper left")
    ax[0].set_title(f"Electricity-heat coupling, {sid}, week from {start}", loc="left")

    ax[1].plot(tt, d["heat_demand"][m], color=INK2, lw=1.3, label="heat demand")
    ax[1].fill_between(
        tt, d["heat_supply"][m], color=ORANGE, alpha=0.5, lw=0, label="heat pump output"
    )
    ax[1].plot(
        tt, d["tes_discharge"][m] - d["tes_charge"][m], color=AQUA, lw=1.1,
        label="tank (+ discharge / - charge)",
    )
    ax[1].axhline(0, color=MUTED, lw=0.6)
    ax[1].set_ylabel("heat kW")
    ax[1].legend(fontsize=8, ncol=3, loc="upper left")

    ax[2].plot(tt, d["tes_soc_kwh"][m] / 1000, color=AQUA, lw=1.4)
    ax[2].set_ylim(bottom=0)
    ax[2].set_ylabel("tank heat\nMWh")
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    return _save(fig, out)


def heat_comparison(
    base: pd.DataFrame,
    heat: pd.DataFrame,
    out: Path,
    battery_kwh: float = 1450,
    n_chargers: int | None = None,
) -> Path | None:
    """Self-consumption and self-sufficiency vs added PV, without and with heat coupling.

    ``base`` and ``heat`` are the kpis.csv of the electricity-only and the heat-coupled
    sweep over the same grid.
    """
    n_chargers = base["n_chargers"].min() if n_chargers is None else n_chargers
    sel = lambda k: (  # noqa: E731
        k[(k["battery_kwh"] == battery_kwh) & (k["n_chargers"] == n_chargers)]
        .sort_values("added_pv_kwp")
        .reset_index(drop=True)
    )
    b = sel(base)
    # one bar per tank size of the heat sweep (tes_scale 0 = heat pumps without a tank)
    groups = [("electricity only", BLUE, b)]
    for scale, color in zip(sorted(heat["tes_scale"].unique()), (YELLOW, ORANGE, AQUA)):
        label = "heat pumps, no tank" if scale == 0 else f"heat pumps + tank x{scale:g}"
        groups.append((label, color, sel(heat[heat["tes_scale"] == scale])))
    pv = sorted(set.intersection(*(set(g["added_pv_kwp"]) for _, _, g in groups)))
    if not pv:
        return None
    groups = [(n, c, g[g["added_pv_kwp"].isin(pv)]) for n, c, g in groups]
    x = np.arange(len(pv))
    w = 0.8 / len(groups)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, title in zip(axes, ("SC", "SS"), ("self-consumption", "self-sufficiency")):
        for i, (name, color, g) in enumerate(groups):
            ax.bar(x + (i - (len(groups) - 1) / 2) * w, g[col], w, color=color, label=name)
        ax.set_xticks(x, [f"{p:g}" for p in pv])
        ax.set_xlabel("added PV (kWp)")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        f"{battery_kwh:g} kWh battery, {n_chargers} chargers", x=0.01, ha="left", fontsize=10
    )
    return _save(fig, out)

