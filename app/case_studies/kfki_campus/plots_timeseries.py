"""Time-series figures: the cleaned input data and simulated scenario weeks.

Style as in plots.py: recessive grid, ink-coloured text, categorical colours in fixed
order (never more than six series per chart), sequential ramps for ordinal levels.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.case_studies.kfki_campus.excel_io import PLOT_MERGE, CampusData  # noqa: E402
from app.case_studies.kfki_campus.plots import (  # noqa: E402
    BLUE,
    CMAP_BLUE,
    CMAP_ORANGE,
    INK,
    INK2,
    MUTED,
    ORANGE,
    SURFACE,
)

AQUA, YELLOW, MAGENTA, GREY = "#1baf7a", "#eda100", "#e87ba4", "#b4b2a9"
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]  # categorical slots 1-5, fixed order
SEASONS = {
    "summer (Jun-Aug)": (6, 7, 8),
    "autumn (Sep-Nov)": (9, 10, 11),
    "winter (Dec-Feb)": (12, 1, 2),
}


def _save(fig, out: Path, dpi: int = 160) -> Path:
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def split_loads(csv: pd.DataFrame, data: CampusData, mapped: set) -> pd.DataFrame:
    """Per-load kW columns (by load id) plus total / metered / unmetered sums."""
    cols = {r.id: csv[r.csv_column] for r in data.loads.itertuples()}
    loads = pd.DataFrame(cols)
    loads["total"] = loads.sum(axis=1)
    ids = [c for c in cols]
    loads["metered"] = loads[[i for i in ids if i in mapped]].sum(axis=1)
    loads["unmetered"] = loads["total"] - loads["metered"]
    return loads


# ------------------------------------------------------------ input data


def input_overview(csv, weather, loads, out: Path) -> Path:
    """Daily campus power split, PV yield per kWp and air temperature over the period."""
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    d = loads[["metered", "unmetered"]].resample("1D").mean() / 1000.0
    axes[0].stackplot(
        d.index,
        d["metered"],
        d["unmetered"],
        colors=[BLUE, GREY],
        labels=["measured buildings (13 meters)", "unmetered rest (E.ON - measured)"],
        lw=0,
    )
    axes[0].set_ylabel("MW (daily mean)")
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    axes[0].set_title(
        "Campus demand, PV potential and weather over the simulated period", loc="left"
    )
    pv = csv["pv_per_kwp"].resample("1D").sum()
    axes[1].fill_between(pv.index, pv, color=ORANGE, alpha=0.9, lw=0)
    axes[1].set_ylabel("PV yield\n(kWh/kWp/day)")
    if weather is not None:
        t = weather["temp_c"].resample("1D")
        axes[2].fill_between(
            t.min().index, t.min(), t.max(), color=BLUE, alpha=0.25, lw=0
        )
        axes[2].plot(t.mean(), color=BLUE, lw=1.2)
    axes[2].set_ylabel("air temperature (degC)\ndaily min / mean / max")
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    return _save(fig, out)


def input_daily_profiles(csv, loads, out: Path) -> Path:
    """Mean daily profile (with inter-quartile band) by season, weekday vs weekend; PV below."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    hour = loads.index.hour
    for j, (name, months) in enumerate(SEASONS.items()):
        sel = loads.index.month.isin(months)
        if not sel.any():  # period does not cover this season
            axes[0, j].axis("off")
            axes[1, j].axis("off")
            continue
        for label, wk, col in (
            ("weekday", loads.index.dayofweek < 5, BLUE),
            ("weekend", loads.index.dayofweek >= 5, ORANGE),
        ):
            s = loads.loc[sel & wk, "total"] / 1000.0
            g = s.groupby(s.index.hour)
            axes[0, j].fill_between(
                range(24),
                g.quantile(0.25),
                g.quantile(0.75),
                color=col,
                alpha=0.2,
                lw=0,
            )
            axes[0, j].plot(range(24), g.mean(), color=col, lw=1.8, label=label)
        pv = csv.loc[sel, "pv_per_kwp"]
        g = pv.groupby(pv.index.hour)
        axes[1, j].fill_between(
            range(24), g.quantile(0.1), g.quantile(0.9), color=ORANGE, alpha=0.2, lw=0
        )
        axes[1, j].plot(range(24), g.mean(), color=ORANGE, lw=1.8)
        axes[0, j].set_title(name, loc="left")
        axes[1, j].set_xlabel("hour of day (UTC+1)")
        axes[1, j].set_xticks(range(0, 24, 4))
    axes[0, 0].set_ylabel("campus demand (MW)\nmean, band = quartiles")
    axes[1, 0].set_ylabel("PV (kW per kWp)\nmean, band = 10-90 %")
    axes[0, 0].legend(fontsize=8)
    return _save(fig, out)


def input_carpets(csv, loads, out: Path) -> Path:
    """Hour-of-day x date maps of campus demand and PV yield."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, series, cmap, title, unit in (
        (axes[0], loads["total"] / 1000.0, CMAP_BLUE, "Campus demand", "MW"),
        (axes[1], csv["pv_per_kwp"], CMAP_ORANGE, "PV output per kWp", "kW/kWp"),
    ):
        grid = (
            series.groupby([series.index.hour, series.index.normalize()])
            .mean()
            .unstack(0)
            .T
        )
        days = grid.columns
        im = ax.imshow(
            grid.to_numpy(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[
                mdates.date2num(days[0]),
                mdates.date2num(days[-1]) + 1,
                -0.5,
                23.5,
            ],
        )
        ax.set_ylabel("hour (UTC+1)")
        ax.set_title(title, loc="left")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label(unit, color=INK2)
        cb.outline.set_visible(False)
    axes[1].xaxis_date()
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    return _save(fig, out)


def input_composition_weeks(
    loads,
    mapped: set,
    out: Path,
    weeks=(("2025-07-07", "summer week"), ("2026-01-12", "winter week")),
    names: dict | None = None,
) -> Path:
    """Stacked hourly demand: the four largest measured buildings, the other measured ones, the unmetered rest."""
    ids = [c for c in loads.columns if c in mapped]
    top = loads[ids].sum().sort_values(ascending=False).index[:4].tolist()
    fig, axes = plt.subplots(
        1, len(weeks), figsize=(6.2 * len(weeks), 4.4), sharey=True
    )
    for ax, (start, name) in zip(np.atleast_1d(axes), weeks):
        w = loads.loc[start : pd.Timestamp(start) + pd.Timedelta(days=7)]
        other = w[[i for i in ids if i not in top]].sum(axis=1)
        layers = [w[i] / 1000 for i in top] + [other / 1000, w["unmetered"] / 1000]
        labels = [(names or {}).get(i, i) for i in top] + [
            "other measured",
            "unmetered rest",
        ]
        ax.stackplot(
            w.index, *layers, colors=CAT[:4] + [MAGENTA, GREY], labels=labels, lw=0
        )
        ax.set_title(f"{name} from {start}", loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    np.atleast_1d(axes)[0].set_ylabel("MW")
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=2, loc="upper left")
    return _save(fig, out)


def input_energy_by_load(
    loads, mapped: set, out: Path, names: dict | None = None
) -> Path:
    """Energy per load over the period: measured vs estimated from the E.ON residual."""
    ids = [c for c in loads.columns if c not in ("total", "metered", "unmetered")]
    own = loads[ids].sum() / 1000.0
    # loads shown as part of another one (PLOT_MERGE): their energy is added to its bar
    extra = pd.Series(0.0, index=own.index)
    for src, dst in PLOT_MERGE.items():
        if src in own.index and dst in own.index:
            extra[dst] += own[src]
            own = own.drop(src)
            extra = extra.drop(src)
    mwh = (own + extra).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 8))
    colours = [BLUE if i in mapped else GREY for i in mwh.index]
    ax.barh(range(len(mwh)), own[mwh.index].values, color=colours, height=0.7)
    ax.barh(  # estimated part added to a measured building
        range(len(mwh)),
        extra[mwh.index].values,
        left=own[mwh.index].values,
        color=GREY,
        height=0.7,
    )
    ax.set_yticks(
        range(len(mwh)), [(names or {}).get(i, i) for i in mwh.index], fontsize=7
    )
    for k, v in enumerate(mwh.values):
        ax.text(v, k, f" {v:,.0f}", va="center", fontsize=6.5, color=INK2)
    ax.set_xlabel("MWh over the period")
    ax.set_title(
        "Energy per load: measured (blue) vs allocated from the E.ON residual (grey)",
        loc="left",
        fontsize=10,
    )
    ax.grid(axis="y", visible=False)
    return _save(fig, out)


def input_measured_buildings(
    loads, mapped: set, out: Path, names: dict | None = None
) -> Path:
    """Small multiples of the measured buildings, 24 h mean power (kW) per day."""
    ids = sorted(
        (c for c in loads.columns if c in mapped), key=lambda c: -loads[c].mean()
    )
    n = len(ids)
    cols = 3
    fig, axes = plt.subplots(
        int(np.ceil(n / cols)),
        cols,
        figsize=(11, 2.3 * int(np.ceil(n / cols))),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, ids):
        d = loads[i].resample("1D").mean()
        ax.plot(d.index, d, color=BLUE, lw=0.7, alpha=0.5)
        ax.plot(d.rolling(7, center=True, min_periods=3).mean(), color=BLUE, lw=1.6)
        label = (names or {}).get(i, i)
        ax.set_title(
            f"{label}  (mean {loads[i].mean():.0f} kW)", loc="left", fontsize=9
        )
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        "Measured buildings: daily mean power (thin) and 7-day mean (bold), kW",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    return _save(fig, out)


def input_ev_profiles(
    csv: pd.DataFrame,
    out: Path,
    n_base: int = 3,
    n_max: int = 9,
    week: str = "2026-01-12",
) -> Path | None:
    """EV charger demand: one week stacked by charger, and the mean weekday profile."""
    cols = sorted(c for c in csv.columns if c.startswith("ev_charger_"))
    if len(cols) < n_max:
        return None
    ev = csv[cols[:n_max]]
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    w = ev.loc[week : pd.Timestamp(week) + pd.Timedelta(days=7)]
    layers = [w[c] for c in cols[:n_base]] + [w[cols[n_base:n_max]].sum(axis=1)]
    labels = [f"charger {i + 1}" for i in range(n_base)] + [
        f"chargers {n_base + 1}-{n_max}"
    ]
    axes[0].stackplot(
        w.index, *layers, colors=CAT[:n_base] + [GREY], labels=labels, lw=0
    )
    axes[0].set_ylabel("kW")
    axes[0].set_title(f"Charging demand, week from {week}", loc="left")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    wd = ev[ev.index.dayofweek < 5]
    for n, colr in ((n_base, BLUE), (n_max, ORANGE)):
        tot = wd[cols[:n]].sum(axis=1)
        g = tot.groupby(tot.index.hour)
        axes[1].fill_between(
            range(24), g.quantile(0.25), g.quantile(0.75), color=colr, alpha=0.2, lw=0
        )
        axes[1].plot(range(24), g.mean(), color=colr, lw=1.8, label=f"{n} chargers")
    axes[1].set_xlabel("hour of day (UTC+1), weekdays")
    axes[1].set_xticks(range(0, 24, 4))
    axes[1].set_ylabel("kW (mean, band = quartiles)")
    axes[1].set_title("Mean weekday profile", loc="left")
    axes[1].legend(fontsize=8)
    return _save(fig, out)


# --------------------------------------------------------- scenario data


def _load_run(details: Path, sid: str) -> dict:
    z = np.load(Path(details) / f"{sid}.npz")
    d = {k: z[k] for k in z.files}
    d["time"] = pd.to_datetime(d["index"], unit="s")
    return d


def scenario_week(
    details: Path, sid: str, start: str, out: Path, days: int = 7
) -> Path | None:
    """Power flows, battery, transformer loading and voltage for one scenario and week."""
    if not (Path(details) / f"{sid}.npz").exists():
        return None
    d = _load_run(details, sid)
    t = d["time"]
    m = (t >= pd.Timestamp(start)) & (t < pd.Timestamp(start) + pd.Timedelta(days=days))
    if not m.any():
        return None
    exists = d["soc_kwh"].max() > 0
    has_bat = exists and (
        d["p_charge"][m].max() + d["p_discharge"][m].max() > 1.0  # kW: used this week
    )
    rows = ["power"] + (["battery", "soc"] if has_bat else []) + ["trafo", "voltage"]
    heights = {"power": 2.4, "battery": 1.0, "soc": 1.0, "trafo": 1.5, "voltage": 1.3}
    fig, axes = plt.subplots(
        len(rows),
        1,
        figsize=(10, sum(heights[r] for r in rows) + 0.8),
        sharex=True,
        gridspec_kw={"height_ratios": [heights[r] for r in rows]},
    )
    ax = dict(zip(rows, np.atleast_1d(axes)))
    tt = t[m]
    ax["power"].plot(tt, d["demand"][m] / 1000, color=INK2, lw=1.2, label="demand")
    ax["power"].plot(tt, d["pv"][m] / 1000, color=ORANGE, lw=1.4, label="PV")
    ax["power"].plot(
        tt,
        (d["p_import"][m] - d["p_export"][m]) / 1000,
        color=BLUE,
        lw=1.4,
        label="grid (+import / -export)",
    )
    ax["power"].axhline(0, color=MUTED, lw=0.6)
    ax["power"].set_ylabel("MW")
    ax["power"].legend(fontsize=8, ncol=3, loc="upper left")
    note = " (battery idle this week)" if exists and not has_bat else ""
    ax["power"].set_title(f"Scenario {sid}, week from {start}{note}", loc="left")
    if has_bat:
        net = (d["p_discharge"][m] - d["p_charge"][m]) / 1000
        ax["battery"].fill_between(tt, net, color=AQUA, lw=0, alpha=0.9)
        ax["battery"].axhline(0, color=MUTED, lw=0.6)
        ax["battery"].set_ylabel("battery MW\n(+ discharge)")
        ax["soc"].plot(tt, d["soc_kwh"][m] / 1000, color=AQUA, lw=1.4)
        ax["soc"].set_ylabel("stored\nenergy MWh")
    for k, name in enumerate(map(str, d["trafo_names"])):
        ax["trafo"].plot(
            tt,
            d["trafo_loading"][m][:, k],
            color=CAT[k % len(CAT)],
            lw=1.1,
            label=name.replace("TR_", ""),
        )
    ax["trafo"].axhline(100, color=MUTED, lw=0.8, ls="--")
    ax["trafo"].set_ylabel("transformer\nloading %")
    ax["trafo"].legend(fontsize=7, ncol=5, loc="upper left")
    ax["voltage"].plot(tt, d["vm_max"][m], color=ORANGE, lw=1.3, label="highest bus")
    ax["voltage"].plot(tt, d["vm_min"][m], color=BLUE, lw=1.3, label="lowest bus")
    ax["voltage"].axhline(1.05, color=MUTED, lw=0.8, ls="--")
    ax["voltage"].set_ylabel("voltage (pu)")
    ax["voltage"].legend(fontsize=7, ncol=2, loc="upper left")
    ax[rows[-1]].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    return _save(fig, out)


def voltage_duration(
    kpis: pd.DataFrame,
    details: Path,
    out: Path,
    battery_kwh: float = 1450,
    n_chargers: int | None = None,  # None = fewest chargers in the sweep
) -> Path | None:
    """Duration curves of the highest and lowest bus voltage for increasing added PV."""
    n_chargers = kpis["n_chargers"].min() if n_chargers is None else n_chargers
    sel = kpis[
        (kpis["battery_kwh"] == battery_kwh) & (kpis["n_chargers"] == n_chargers)
    ].sort_values("added_pv_kwp")
    sel = sel[[(Path(details) / f"{s}.npz").exists() for s in sel["scenario"]]]
    if sel.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, r in enumerate(sel.itertuples()):
        z = np.load(Path(details) / f"{r.scenario}.npz")
        c = CMAP_BLUE(0.25 + 0.75 * k / max(len(sel) - 1, 1))
        x = np.linspace(0, 100, len(z["vm_max"]))
        axes[0].plot(
            x,
            np.sort(z["vm_max"])[::-1],
            color=c,
            lw=1.4,
            label=f"{r.added_pv_kwp:g} kWp",
        )
        axes[1].plot(x, np.sort(z["vm_min"]), color=c, lw=1.4)
    axes[0].axhline(1.05, color=MUTED, lw=0.8, ls="--")
    axes[0].set_ylabel("highest bus voltage (pu)")
    axes[0].set_title(
        f"Highest voltage, sorted over time (battery {battery_kwh:g} kWh)",
        loc="left",
        fontsize=10,
    )
    axes[1].set_ylabel("lowest bus voltage (pu)")
    axes[1].set_title("Lowest voltage, sorted over time", loc="left", fontsize=10)
    for a in axes:
        a.set_xlabel("share of time (%)")
    axes[0].legend(
        title="added PV", fontsize=7, title_fontsize=8, ncol=2, loc="upper right"
    )
    return _save(fig, out)
