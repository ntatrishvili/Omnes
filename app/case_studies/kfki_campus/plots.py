"""Figures for the KFKI scenario sweep (static PNGs for the report).

Style follows the dataviz conventions: one-hue sequential ramps for magnitudes
(blue for SC/SS, orange for violations), categorical slots in fixed order for
the compared scenarios, hairline recessive grids, text in ink colours (never
the series colour), values printed directly in heatmap cells.
"""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GREY = "#b4b2a9"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fde3d6", "#f8b99c", "#f08e63", "#d9622f", "#a8431a", "#6e2a0f"]
CMAP_BLUE = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
CMAP_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ORANGE_RAMP)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.titlesize": 11,
        "axes.titlecolor": INK,
        "legend.frameon": False,
    }
)


def _panel_levels(levels: list) -> list:
    """Charger counts to show as panels: all if <= 4, else lowest, middle and highest."""
    levels = sorted(levels)
    if len(levels) <= 4:
        return levels
    return [levels[0], levels[len(levels) // 2], levels[-1]]


def _heat(df, metric, title, cmap, fmt, out, vmax=None, note="", vmin=0.0):
    chargers = _panel_levels(list(df["n_chargers"].unique()))
    pvs = sorted(df["added_pv_kwp"].unique())
    bats = sorted(df["battery_kwh"].unique())
    vmax = vmax if vmax is not None else max(float(np.nanmax(df[metric])), 1e-12)
    panel_w = max(3.4, 0.62 * len(pvs) + 0.8)  # keep cell labels legible on wide grids
    fig, axes = plt.subplots(
        1,
        len(chargers),
        figsize=(panel_w * len(chargers) + 1, 3.9),
        sharey=True,
        squeeze=False,
    )
    for ax, ev in zip(axes[0], chargers):
        sub = (
            df[df["n_chargers"] == ev]
            .pivot(index="battery_kwh", columns="added_pv_kwp", values=metric)
            .reindex(index=bats, columns=pvs)
        )
        im = ax.imshow(
            sub.to_numpy(),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.grid(False)
        ax.set_xticks(range(len(pvs)), [f"{p:g}" for p in pvs])
        ax.set_yticks(range(len(bats)), [f"{b:g}" for b in bats])
        ax.set_title(f"{ev} chargers", color=INK)
        ax.set_xlabel("added PV (kWp)")
        for i, b in enumerate(bats):
            for j, p in enumerate(pvs):
                v = sub.loc[b, p]
                if pd.notna(v):
                    dark = ((v - vmin) / max(vmax - vmin, 1e-12)) > 0.55
                    ax.text(
                        j,
                        i,
                        fmt(v),
                        ha="center",
                        va="center",
                        fontsize=7 if len(pvs) > 6 else 8,
                        color="#ffffff" if dark else INK,
                    )
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0][0].set_ylabel("battery (kWh)")
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    if note:
        fig.text(0.01, 0.01, note, color=MUTED, fontsize=8)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def ev_effect(df: pd.DataFrame, out: Path, battery_kwh: float = 1450.0) -> Path:
    """KPIs against the number of chargers, one line per added-PV level (fixed battery)."""
    sub = df[df["battery_kwh"] == battery_kwh]
    pvs = [
        p for p in (0.0, 500.0, 1000.0, 1500.0, 2000.0) if p in set(sub["added_pv_kwp"])
    ]
    panels = [
        ("SS", "self-sufficiency SS", "{:.3f}"),
        ("SC", "self-consumption SC", "{:.3f}"),
        ("v_min_pu", "lowest bus voltage (pu)", "{:.3f}"),
        ("trafo_max_loading_pct", "highest transformer loading (%)", "{:.0f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (col, label, fmt) in zip(axes, panels):
        for k, p in enumerate(pvs):
            s = sub[sub["added_pv_kwp"] == p].sort_values("n_chargers")
            c = CMAP_BLUE(0.3 + 0.7 * k / max(len(pvs) - 1, 1))
            ax.plot(
                s["n_chargers"],
                s[col],
                color=c,
                lw=1.8,
                marker="o",
                ms=3.5,
                label=f"{p:g} kWp",
            )
        ax.set_xlabel("EV chargers")
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xticks(sorted(sub["n_chargers"].unique()))
    axes[0].legend(title="added PV", fontsize=7, title_fontsize=8)
    fig.suptitle(
        f"Effect of the number of EV chargers (battery {battery_kwh:g} kWh)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def pick_scenarios(df: pd.DataFrame) -> tuple[str, str]:
    """Baseline (no added PV, today's chargers, battery nearest the existing 1450 kWh) and
    the highest-voltage scenario."""
    base = df[
        (df["added_pv_kwp"] == df["added_pv_kwp"].min())
        & (df["n_chargers"] == df["n_chargers"].min())
    ]
    base = base.iloc[(base["battery_kwh"] - 1450).abs().argsort()].iloc[0]
    worst = df.sort_values(["viol_frac", "v_max_pu"], ascending=False).iloc[0]
    return base["scenario"], worst["scenario"]


def _load(details: Path, sid: str) -> dict:
    z = np.load(details / f"{sid}.npz")
    d = {k: z[k] for k in z.files}
    d["time"] = pd.to_datetime(d["index"], unit="s")
    return d


def make_figures(
    out_dir: Path, v_max: float = 1.1, v_strict: float = 1.05
) -> list[Path]:
    """Create all figures from ``out_dir/kpis.csv`` (+ ``details/``); return their paths."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out_dir / "kpis.csv")
    made = []

    _heat(
        df,
        "SC",
        "Self-consumption (SC) of PV energy",
        CMAP_BLUE,
        lambda v: f"{v:.2f}",
        fig_dir / "heatmap_SC.png",
        vmax=1.0,
    )
    _heat(
        df,
        "SS",
        "Self-sufficiency (SS) of campus demand",
        CMAP_BLUE,
        lambda v: f"{v:.2f}",
        fig_dir / "heatmap_SS.png",
    )
    _heat(
        df,
        "viol_frac",
        "Voltage violations (share of bus-hours outside 0.9-1.1 pu)",
        CMAP_ORANGE,
        lambda v: f"{100 * v:.2f}%",
        fig_dir / "heatmap_violations.png",
        vmax=max(float(df["viol_frac"].max()), 1e-6),
        note=(
            "Empty (white) = no violations in any scenario."
            if df["viol_frac"].max() == 0
            else ""
        ),
    )
    _heat(
        df,
        "v_max_pu",
        "Highest bus voltage reached (pu)",
        CMAP_ORANGE,
        lambda v: f"{v:.3f}",
        fig_dir / "heatmap_vmax.png",
        vmin=1.0,  # nominal voltage; otherwise every cell looks the same
        vmax=max(float(df["v_max_pu"].max()), 1.0 + 1e-6),
    )
    made += [
        fig_dir / f
        for f in (
            "heatmap_SC.png",
            "heatmap_SS.png",
            "heatmap_violations.png",
            "heatmap_vmax.png",
        )
    ]

    # SC vs SS trade-off, colour = added PV (ordinal blue steps), size = battery
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    pvs = sorted(df["added_pv_kwp"].unique())
    steps = np.linspace(0.25, 1.0, len(pvs))
    for p, s in zip(pvs, steps):
        # large markers first so the small ones (less battery) stay visible on top
        sub = df[df["added_pv_kwp"] == p].sort_values("battery_kwh", ascending=False)
        ax.scatter(
            sub["SS"],
            sub["SC"],
            s=20 + sub["battery_kwh"] / 40,
            color=CMAP_BLUE(s),
            edgecolor=SURFACE,
            linewidth=1.2,
            label=f"{p:g} kWp",
        )
    ax.set_xlabel("self-sufficiency SS")
    ax.set_ylabel("self-consumption SC")
    ax.set_title("SC vs SS across scenarios", loc="left")
    pv_legend = ax.legend(
        title="added PV (colour)",
        fontsize=8,
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    ax.add_artist(pv_legend)
    # separate legend for the exact battery sizes: grey markers drawn at the same area
    sizes = sorted(df["battery_kwh"].unique())
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            markersize=float(np.sqrt(20 + b / 40)),
            markerfacecolor=GREY,
            markeredgecolor=INK2,
            label=f"{b:g} kWh" if b else "no battery",
        )
        for b in sizes
    ]
    ax.legend(
        handles=handles,
        title="battery (marker size)",
        fontsize=8,
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.52),
        labelspacing=1.0,
        borderpad=0.8,
    )
    fig.savefig(fig_dir / "sc_vs_ss.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    made.append(fig_dir / "sc_vs_ss.png")
    if df["n_chargers"].nunique() > 1:
        made.append(ev_effect(df, fig_dir / "ev_effect.png"))

    details = out_dir / "details"
    if details.exists():
        base_id, worst_id = pick_scenarios(df)
        runs = [
            (base_id, BLUE, "baseline"),
            (worst_id, ORANGE, "highest-voltage scenario"),
        ]
        data = {
            sid: _load(details, sid)
            for sid, _, _ in runs
            if (details / f"{sid}.npz").exists()
        }
        # the dispatch week is only informative with a battery: take the highest-voltage
        # scenario that has one
        with_bat = df[df["battery_kwh"] > 0].sort_values(
            ["viol_frac", "v_max_pu"], ascending=False
        )
        disp_id = with_bat.iloc[0]["scenario"] if len(with_bat) else worst_id
        if disp_id not in data and (details / f"{disp_id}.npz").exists():
            data[disp_id] = _load(details, disp_id)
        if len(data) >= 2:
            made += _detail_figs(data, runs, fig_dir, v_max, v_strict, disp_id)
    return made


def _detail_figs(data, runs, fig_dir, v_max, v_strict, dispatch_id) -> list[Path]:
    made = []
    # daily voltage envelope over the year
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for sid, col, label in runs:
        d = data[sid]
        s_max = pd.Series(d["vm_max"], index=d["time"]).resample("1D").max()
        s_min = pd.Series(d["vm_min"], index=d["time"]).resample("1D").min()
        ax.plot(s_max.index, s_max, color=col, lw=1.4, label=f"{label} ({sid})")
        ax.plot(s_min.index, s_min, color=col, lw=1.4)
    for lim, name in ((v_max, "1.10 pu limit"), (v_strict, "1.05 pu")):
        ax.axhline(lim, color=MUTED, lw=0.8, ls="--")
        ax.text(ax.get_xlim()[0], lim, f" {name}", color=INK2, fontsize=8, va="bottom")
    ax.set_ylabel("bus voltage (pu), daily max / min")
    ax.set_title("Voltage envelope over the simulated period", loc="left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.savefig(fig_dir / "voltage_envelope.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    made.append(fig_dir / "voltage_envelope.png")

    # transformer loading duration curve (most loaded transformer each step)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for sid, col, label in runs:
        tl = np.nanmax(data[sid]["trafo_loading"], axis=1)
        ax.plot(
            np.linspace(0, 100, len(tl)),
            np.sort(tl)[::-1],
            color=col,
            lw=1.6,
            label=f"{label} ({sid})",
        )
    ax.set_xlabel("share of time (%)")
    ax.set_ylabel("loading of most loaded transformer (%)")
    ax.set_title("Transformer loading duration curve", loc="left")
    ax.legend(fontsize=8)
    fig.savefig(fig_dir / "trafo_duration.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    made.append(fig_dir / "trafo_duration.png")

    # one summer week of the highest-voltage scenario: power flows and battery state
    sid = dispatch_id
    d = data[sid]
    t = d["time"]
    start = pd.Timestamp(f"{t[0].year}-06-17")
    m = (t >= start) & (t < start + pd.Timedelta(days=7))
    if m.any():
        fig, (a1, a2) = plt.subplots(
            2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
        )
        a1.plot(t[m], d["demand"][m] / 1000, color=INK2, lw=1.2, label="demand")
        a1.plot(t[m], d["pv"][m] / 1000, color=ORANGE, lw=1.4, label="PV")
        a1.plot(
            t[m],
            (d["p_import"][m] - d["p_export"][m]) / 1000,
            color=BLUE,
            lw=1.4,
            label="grid (+import / -export)",
        )
        a1.set_ylabel("MW")
        a1.legend(fontsize=8, ncol=3, loc="upper left")
        a1.set_title(f"Summer week, scenario {sid}", loc="left")
        a2.plot(t[m], d["soc_kwh"][m] / 1000, color=BLUE, lw=1.4)
        a2.set_ylabel("battery energy (MWh)")
        a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        fig.savefig(fig_dir / "dispatch_week.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        made.append(fig_dir / "dispatch_week.png")
    return made
