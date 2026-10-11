"""Publication figures for the conference paper (IEEE column widths, vector PDF).

Reads the finished sweep (``results/kfki``) and the heat demo (``results_heat_demo``);
nothing is re-simulated.  One style for every figure: project palette, 7 pt text,
panel labels (a), (b), ...  Semantic colours are fixed across figures:
PV orange, grid/battery blue, EV green, heat purple, other demand grey.

Run:  python -m app.case_studies.kfki_campus.paper_figures [--out paper/figures]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from app.case_studies.kfki_campus.plots import (  # noqa: E402
    AXIS,
    BLUE,
    CMAP_BLUE,
    CMAP_ORANGE,
    GRID,
    INK,
    INK2,
    MUTED,
    ORANGE,
)

GREEN, PURPLE = "#1a9e77", "#5b3fa8"
LIGHT = "#f1f0ea"
COL, FULL = 3.5, 7.16  # IEEE single / double column width (in)
FONT = "DejaVu Sans"  # switch to a serif family here to match the body text

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": FONT,
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "pdf.fonttype": 42,
    }
)


def _label(ax, tag, x=-0.02, y=1.04):
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK)


def _save(fig, out: Path, name: str) -> Path:
    path = out / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- Fig. 1
def _box(ax, x, y, w, h, text, fc=LIGHT, ec=AXIS, bold=False, fs=6.5, color=INK):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.012", fc=fc, ec=ec, lw=0.7
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
        fontweight="bold" if bold else "normal", color=color, linespacing=1.15,
    )


def _arrow(ax, p, q, color=INK2, style="-|>", lw=0.8, rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p, q, arrowstyle=style, mutation_scale=6, color=color, lw=lw, ls=ls,
            connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
        )
    )


def _marker(ax, x, y, kind, fs=5.5):
    if kind == "pv_new":
        ax.plot(x, y, "s", mfc="white", mec=ORANGE, mew=1.0, ms=fs)
    elif kind == "pv_old":
        ax.plot(x, y, "s", color=ORANGE, ms=fs)
    elif kind == "bat":
        ax.plot(x, y, "D", color=BLUE, ms=fs)
    elif kind == "ev":
        ax.plot(x, y, "^", mfc="white", mec=GREEN, mew=1.0, ms=fs)
    elif kind == "hp":
        ax.plot(x, y, "h", mfc="white", mec=PURPLE, mew=1.0, ms=fs)


def fig_system(out: Path) -> Path:
    """Fig. 1: the campus network as modelled (full width)."""
    fig = plt.figure(figsize=(FULL, 2.3))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1)
    ax.axis("off")

    _box(ax, 0.41, 0.82, 0.12, 0.07, "Slack\n10 kV grid", fs=6)
    ax.plot([0.47, 0.47], [0.82, 0.76], color=INK, lw=0.8)
    ax.plot([0.03, 0.97], [0.76, 0.76], color=INK, lw=3)
    ax.text(0.03, 0.79, "10 kV bus", fontsize=6, color=INK2)
    buses = [  # name, centre x, feeders, mean MW, markers
        ("T", 0.12, 7, 0.20, ["pv_new", "ev", "hp"]),
        ("3T", 0.31, 10, 0.13, ["ev"]),
        ("4T", 0.50, 5, 0.10, ["pv_new", "pv_new", "ev", "hp", "hp", "bat"]),
        ("6T", 0.69, 7, 0.23, ["pv_old", "pv_new", "ev", "hp"]),
        ("10T", 0.88, 4, 0.19, ["pv_new"]),
    ]
    for name, cx, n, mw, marks in buses:
        ax.plot([cx, cx], [0.76, 0.62], color=INK, lw=0.8)
        ax.add_patch(plt.Circle((cx, 0.69), 0.020, fc="white", ec=INK, lw=0.8))
        ax.add_patch(plt.Circle((cx, 0.665), 0.020, fc="white", ec=INK, lw=0.8))
        ax.text(cx + 0.035, 0.69, f"{name}\n1 MVA", fontsize=5.5, color=INK2, va="center")
        half = 0.0075 * n + 0.02
        ax.plot([cx - half, cx + half], [0.60, 0.60], color=INK, lw=2.4)
        xs = np.linspace(cx - half + 0.008, cx + half - 0.008, n)
        for x in xs:
            ax.plot([x, x], [0.60, 0.50], color=AXIS, lw=0.6)
            ax.plot(x, 0.50, "o", color=GRID if False else MUTED, ms=2.2)
        ax.text(cx, 0.385, f"{n} feeders\n{mw:.2f} MW avg", ha="center",
                fontsize=5.5, color=INK2, va="top")
        for k, m in enumerate(marks):
            _marker(ax, cx - (len(marks) - 1) * 0.013 + k * 0.026, 0.455, m)
    ax.text(0.5, 0.215, "LV level: one radial feeder (bus + line) per consumer,\n33 consumers, 5 x 1 MVA 10/0.4 kV transformers",
            ha="center", fontsize=5.8, color=INK2)
    legend = [
        ("pv_old", "existing PV"), ("pv_new", "PV roof (candidate)"),
        ("bat", "NaS battery"), ("ev", "EV charger site"), ("hp", "heat pump"),
    ]
    x0 = 0.02
    for kind, txt in legend:
        _marker(ax, x0, 0.10, kind)
        ax.text(x0 + 0.012, 0.10, txt, fontsize=6, va="center", color=INK2)
        x0 += 0.035 + 0.0058 * len(txt) + 0.03

    return _save(fig, out, "fig1_system")


# --------------------------------------------------------------------------- Fig. 2
def fig_workflow(out: Path) -> Path:
    """Fig. 2: coupled optimisation-simulation workflow (single column)."""
    fig, ax = plt.subplots(figsize=(COL, 3.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, 0.03, 0.88, 0.94, 0.10, "Measured data: meters, weather, EV profiles\n"
         "hourly, 6672 steps, aligned by the TimeSet", fs=6.2, fc="white")
    _box(ax, 0.03, 0.66, 0.94, 0.14, "OMNES model (internal representation)\n"
         "entities, quantities, topology, parameters", fs=6.8, bold=True,
         fc="#e4eefb", ec=BLUE)
    _arrow(ax, (0.5, 0.88), (0.5, 0.80))
    # two backends
    _box(ax, 0.03, 0.33, 0.40, 0.20, "Dispatch backend\nPuLP + CBC (LP/MILP)\nrolling horizon\n24 h + 24 h look-ahead",
         fs=6.1, fc="white", ec=BLUE)
    _box(ax, 0.57, 0.33, 0.40, 0.20, "Simulation backend\nPandapowerConverter\nNewton-Raphson,\nper-step power flow",
         fs=6.1, fc="white", ec=BLUE)
    _arrow(ax, (0.25, 0.66), (0.25, 0.53))
    _arrow(ax, (0.75, 0.66), (0.75, 0.53))
    _arrow(ax, (0.43, 0.43), (0.57, 0.43), color=BLUE, lw=1.2)
    ax.text(0.5, 0.455, "schedules", ha="center", fontsize=5.2, color=BLUE)
    # heat branch
    _box(ax, 0.03, 0.10, 0.40, 0.14, "Heat chain\nHeatPump - Transducer\n- tank - HeatDemand",
         fs=5.8, fc="#efeaf8", ec=PURPLE)
    _arrow(ax, (0.25, 0.24), (0.25, 0.33), color=PURPLE, style="<|-|>")
    ax.text(0.27, 0.285, "heat balance", fontsize=5.4, color=PURPLE, va="center")
    _box(ax, 0.57, 0.10, 0.40, 0.14, "KPIs: SC, SS, voltage,\ntransformer / line loading",
         fs=6.1, fc="white", ec=INK2)
    _arrow(ax, (0.75, 0.33), (0.75, 0.24))
    ax.text(0.5, 0.01, "scenario sweep: 8 PV x 5 battery x 7 charger levels", ha="center",
            fontsize=5.8, color=INK2)
    return _save(fig, out, "fig2_workflow")


# --------------------------------------------------------------------------- Fig. 3
def fig_sc_ss(out: Path, kpis: Path) -> Path:
    """Fig. 3: (a) SC vs SS, (b) network limits, for the 40 PV x battery scenarios.

    Colour = added PV, marker area = total battery capacity; lines join the battery sizes
    of one PV level (small to large). Chargers are fixed at today's number (no effect).
    """
    from matplotlib.lines import Line2D

    df = pd.read_csv(kpis)
    df = df[df["n_chargers"] == df["n_chargers"].min()]
    pvs = sorted(df["added_pv_kwp"].unique())
    bats = sorted(df["battery_kwh"].unique())
    colours = {p: CMAP_BLUE(0.2 + 0.8 * i / (len(pvs) - 1)) for i, p in enumerate(pvs)}

    def area(b):  # marker area (pt^2), grows with battery capacity
        return 9 + 0.016 * b

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 3.15), gridspec_kw={"wspace": 0.2})

    def draw(ax, xcol, ycol):
        for p in pvs:
            s = df[df["added_pv_kwp"] == p].sort_values("battery_kwh")
            ax.plot(s[xcol], s[ycol], color=colours[p], lw=0.7, alpha=0.7, zorder=1)
        for _, r in df.sort_values("battery_kwh", ascending=False).iterrows():
            ax.scatter(
                r[xcol], r[ycol], s=area(r["battery_kwh"]),
                facecolor=(*colours[r["added_pv_kwp"]][:3], 0.45),
                edgecolor=colours[r["added_pv_kwp"]], linewidth=0.8, zorder=2,
            )
        ax.grid(True, color=GRID)
        ax.spines[["top", "right"]].set_visible(False)

    def pick(pv, bat):
        return df[(df.added_pv_kwp == pv) & (df.battery_kwh == bat)].iloc[0]

    def note(ax, r, xcol, ycol, txt, dx, dy, ha="left"):
        ax.annotate(
            txt, (r[xcol], r[ycol]), xytext=(r[xcol] + dx, r[ycol] + dy), fontsize=6,
            color=INK2, ha=ha, va="center", zorder=5,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkB=2),
        )

    # (a) energy balance
    draw(a, "SS", "SC")
    a.set_xlabel("self-sufficiency SS")
    a.set_ylabel("self-consumption SC")
    a.set_xlim(-0.015, df["SS"].max() + 0.02)
    a.set_ylim(df["SC"].min() - 0.03, 1.02)
    note(a, pick(2000, 0), "SS", "SC", "2000 kWp, no battery", -0.012, -0.015, "right")
    note(a, pick(2000, 5800), "SS", "SC", "5.8 MWh restores SC", -0.02, -0.07, "right")
    _label(a, "(a)", x=-0.14, y=1.03)

    # (b) network limits
    draw(b, "v_max_pu", "trafo_max_loading_pct")
    b.axvline(1.05, color=ORANGE, lw=0.7, ls="--", zorder=0)
    b.text(1.0505, 43.5, "1.05 pu", color=ORANGE, fontsize=6, ha="left", va="bottom")
    b.set_xlabel("highest bus voltage (pu)")
    b.set_ylabel("highest transformer loading (%)")
    b.set_ylim(40, 98)
    note(b, pick(2000, 2900), "v_max_pu", "trafo_max_loading_pct",
         "2000 kWp, 2.9 MWh:\ntransformer 4/T at 92 %", -0.007, 2, "right")
    note(b, pick(2000, 5800), "v_max_pu", "trafo_max_loading_pct",
         "5.8 MWh: voltage\nrise at battery bus", 0.0, 9, "center")
    note(b, pick(0, 1450), "v_max_pu", "trafo_max_loading_pct",
         "today", 0.004, 8, "left")
    _label(b, "(b)", x=-0.14, y=1.03)

    pv_leg = fig.legend(
        [Line2D([0], [0], marker="o", ls="", ms=5, mfc=colours[p], mec=colours[p])
         for p in pvs],
        [f"{p:g}" for p in pvs], title="added PV (kWp), colour", ncol=8, frameon=False,
        loc="lower left", bbox_to_anchor=(0.06, -0.14), columnspacing=0.9,
        handletextpad=0.2, alignment="left",
    )
    pv_leg.get_title().set_fontsize(6.5)
    fig.add_artist(pv_leg)
    bat_leg = fig.legend(
        [Line2D([0], [0], marker="o", ls="", ms=np.sqrt(area(x)), mfc=MUTED, mec=MUTED)
         for x in bats],
        [f"{x / 1000:g}" if x else "none" for x in bats],
        title="battery, total (MWh), size", ncol=5, frameon=False, loc="lower left",
        bbox_to_anchor=(0.56, -0.14), columnspacing=0.9, handletextpad=0.2,
        alignment="left",
    )
    bat_leg.get_title().set_fontsize(6.5)
    return _save(fig, out, "fig3_sc_ss")


# --------------------------------------------------------------------------- Fig. 4
def fig_heat(out: Path, details: Path) -> Path:
    """Fig. 4: electricity-heat coupling over one October week (2000 kWp, 1.45 MWh)."""
    z = np.load(details)
    d = {k: z[k] for k in z.files}
    t = pd.to_datetime(d["index"], unit="s")
    fig, ax = plt.subplots(3, 1, figsize=(COL, 4.1), sharex=True,
                           gridspec_kw={"height_ratios": [2.0, 1.6, 1.0]})
    base = (d["demand"] - d["p_hp"]) / 1000
    ax[0].fill_between(t, base, color=MUTED, alpha=0.35, lw=0, label="other demand")
    ax[0].fill_between(t, base, base + d["p_hp"] / 1000, color=PURPLE, alpha=0.85,
                       lw=0, label="heat pumps")
    ax[0].plot(t, d["pv"] / 1000, color=ORANGE, lw=1.0, label="PV")
    ax[0].plot(t, (d["p_import"] - d["p_export"]) / 1000, color=BLUE, lw=0.9,
               label="grid")
    ax[0].set_ylabel("electric (MW)")
    ax[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.27), handlelength=1.2,
                 columnspacing=0.9, frameon=False)
    ax[1].plot(t, d["heat_demand"], color=INK2, lw=0.9, label="heat demand")
    ax[1].fill_between(t, d["heat_supply"], color=PURPLE, alpha=0.4, lw=0,
                       label="heat-pump output")
    ax[1].plot(t, d["tes_discharge"] - d["tes_charge"], color=GREEN, lw=0.8,
               label="tank (+dis / -ch)")
    ax[1].axhline(0, color=MUTED, lw=0.5)
    ax[1].set_ylabel("heat (kW)")
    ax[1].legend(ncol=3, loc="upper left", handlelength=1.2, columnspacing=0.9,
                 frameon=False, fontsize=5.8)
    ax[2].plot(t, d["tes_soc_kwh"] / 1000, color=GREEN, lw=1.0)
    ax[2].set_ylim(bottom=0)
    ax[2].set_ylabel("tank (MWh)")
    ax[2].xaxis.set_major_locator(mdates.DayLocator())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%a"))
    for k, a in enumerate(ax):
        a.grid(True, color=GRID)
        a.spines[["top", "right"]].set_visible(False)
        _label(a, f"({'abc'[k]})", x=-0.17, y=1.0)
    fig.align_ylabels(ax)
    return _save(fig, out, "fig4_heat")


def main():
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=root / "paper" / "figures")
    ap.add_argument("--kpis", type=Path, default=root / "results" / "kfki" / "kpis.csv")
    ap.add_argument(
        "--heat-details", type=Path,
        default=root / "results_heat_demo" / "heat" / "details"
        / "pv2000_bat1450_ev3_hp1.npz",
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for p in (
        fig_system(a.out), fig_sc_ss(a.out, a.kpis),
        fig_heat(a.out, a.heat_details),
    ):
        print(p)


if __name__ == "__main__":
    main()




