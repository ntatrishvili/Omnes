"""Network structure figures: a schematic single-line layout built from network.xlsx.

Layout: 10 kV bus on top, one transformer per LV busbar, one drop per feeder with the
building label underneath. Node area scales with the mean load, markers show the
assets (existing and candidate PV, battery, EV sites, heat pumps). The same layout is
reused to colour buses by simulated voltage for chosen scenarios.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from app.case_studies.kfki_campus.excel_io import (  # noqa: E402
    PLOT_MERGE,
    CampusData,
    bus_to_trafo,
    display_names,
    merged_bus,
)
from app.case_studies.kfki_campus.plots import (  # noqa: E402
    AXIS,
    BLUE,
    INK,
    INK2,
    MUTED,
    ORANGE,
    SURFACE,
)

AQUA, VIOLET, GREY = "#1baf7a", "#4a3aa7", "#b4b2a9"
# diverging blue (low) - neutral - red (high) around 1.0 pu, neutral midpoint grey
CMAP_V = LinearSegmentedColormap.from_list(
    "volt", ["#184f95", "#5598e7", "#f0efec", "#ec835a", "#a52a2a"]
)

Y_MV, Y_TRAFO, Y_BAR, Y_NODE, Y_LABEL = 0.0, -1.0, -2.0, -3.4, -3.75


@dataclass
class Layout:
    x: dict  # bus id -> x position (feeder terminals, LV bar centres, MV centre)
    bars: dict  # LV bus -> (x0, x1)
    trafo_x: dict  # transformer id -> x
    trafo_lv: dict  # transformer id -> LV bus
    terminals: list  # feeder terminal bus ids in drawing order
    mv: tuple  # (x0, x1)
    mv_bus: str
    group_of: dict  # terminal -> LV bus


def build_layout(data: CampusData) -> Layout:
    """Tree layout: feeders evenly spaced, groups separated by a gap."""
    lines = data.lines
    trafos = data.transformers
    # the 10 kV busbar: the common parent of the transformers' cell buses (or their own
    # HV bus when the transformers hang directly on it)
    mv_parent = dict(zip(lines["to_bus"], lines["from_bus"]))
    first_hv = trafos["from_bus"].iloc[0]
    mv_bus = mv_parent.get(first_hv, first_hv)
    x, bars, trafo_x, trafo_lv, terminals, group_of = {}, {}, {}, {}, [], {}
    cursor = 0.0
    hidden = {  # feeders drawn as part of another building (PLOT_MERGE)
        r.bus for r in data.loads.itertuples() if r.id in PLOT_MERGE
    }
    for t in trafos.itertuples():
        kids = lines.loc[lines["from_bus"] == t.to_bus, "to_bus"].tolist()
        kids = [k for k in kids if k not in hidden]
        xs = []
        for k in kids:
            x[k] = cursor
            xs.append(cursor)
            terminals.append(k)
            group_of[k] = t.to_bus
            cursor += 1.0
        bars[t.to_bus] = (min(xs) - 0.35, max(xs) + 0.35)
        x[t.to_bus] = float(np.mean(xs))
        trafo_x[t.id] = x[t.to_bus]
        trafo_lv[t.id] = t.to_bus
        cursor += 1.4
    x0, x1 = min(trafo_x.values()), max(trafo_x.values())
    x[mv_bus] = (x0 + x1) / 2
    return Layout(
        x, bars, trafo_x, trafo_lv, terminals, (x0 - 0.8, x1 + 0.8), mv_bus, group_of
    )


def mean_kw_by_bus(data: CampusData, csv: pd.DataFrame) -> dict:
    """Mean kW per drawn bus; merged loads are added to the bus they are drawn with."""
    out = {}
    for df in (data.loads, data.heat_pumps):
        for r in df.itertuples():
            bus = merged_bus(data, r.bus)
            out[bus] = out.get(bus, 0.0) + float(csv[r.csv_column].mean())
    return out


def _assets_by_bus(data: CampusData) -> dict:
    a: dict = {}
    for bus in data.pv["bus"]:
        a.setdefault(bus, []).append("pv")
    for bus in data.pv_sites["bus"]:
        a.setdefault(bus, []).append("pv_candidate")
    for bus in data.ev_sites["bus"]:
        a.setdefault(bus, []).append("ev")
    for bus in data.battery["bus"]:
        a.setdefault(bus, []).append("battery")
    for bus in data.heat_pumps["bus"]:
        a.setdefault(bus, []).append("hp")
    return a


_GLYPH = {
    "pv": dict(marker="s", facecolor=ORANGE, edgecolor=ORANGE),
    "pv_candidate": dict(marker="s", facecolor="none", edgecolor=ORANGE),
    "ev": dict(marker="^", facecolor="none", edgecolor=AQUA),
    "battery": dict(marker="D", facecolor=BLUE, edgecolor=BLUE),
    "hp": dict(marker="h", facecolor="none", edgecolor=VIOLET),
}


def _labels(data: CampusData) -> dict:
    return display_names(data)[1]


def pretty_bus(bus: str, names: dict) -> str:
    """English name of any bus: consumer/battery label, '10 kV bus' or 'LV bus T'."""
    if bus in names:
        return names[bus]
    if bus.startswith("MV_"):
        return "10 kV bus"
    if bus.startswith("LV_"):
        return f"LV bus {bus[3:]}"
    return bus


def draw_network(
    ax,
    lay: Layout,
    data: CampusData,
    mean_kw: dict,
    values: dict | None = None,
    norm=None,
    show_assets: bool = True,
    metered_buses: set | None = None,
    label_size: float = 6.3,
):
    """Draw the schematic. ``values`` (bus -> number) colours buses with CMAP_V."""
    labels = _labels(data)
    trafo_of = bus_to_trafo(data)
    colour = (
        (lambda b: CMAP_V(norm(values[b])) if b in values else GREY)
        if values is not None
        else None
    )

    def line_colour(bus, default):
        return colour(bus) if colour else default

    # MV bar and external grid
    x0, x1 = lay.mv
    ax.plot(
        [x0, x1],
        [Y_MV, Y_MV],
        color=line_colour(lay.mv_bus, INK),
        lw=4,
        solid_capstyle="butt",
    )
    gx = lay.x[lay.mv_bus]
    ax.plot([gx, gx], [Y_MV, 0.55], color=INK2, lw=1)
    ax.scatter(
        [gx], [0.7], marker="s", s=150, facecolor=SURFACE, edgecolor=INK, zorder=3
    )
    ax.text(gx, 0.7, "x", ha="center", va="center", fontsize=8, color=INK)
    ax.text(gx + 0.45, 0.7, "external grid 10 kV", fontsize=7, va="center", color=INK2)

    for tid, tx in lay.trafo_x.items():
        lv = lay.trafo_lv[tid]
        ax.plot([tx, tx], [Y_MV, Y_TRAFO + 0.3], color=INK2, lw=1)
        ax.plot([tx, tx], [Y_TRAFO - 0.3, Y_BAR], color=INK2, lw=1)
        for dy in (0.12, -0.12):  # two overlapping circles = transformer symbol
            ax.scatter(
                [tx],
                [Y_TRAFO + dy],
                s=230,
                facecolor="none",
                edgecolor=INK,
                lw=1.1,
                zorder=3,
            )
        ax.text(
            tx + 0.5,
            Y_TRAFO,
            tid.replace("TR_", "") + "\n1 MVA",
            fontsize=6.5,
            va="center",
            color=INK2,
        )
        b0, b1 = lay.bars[lv]
        ax.plot(
            [b0, b1],
            [Y_BAR, Y_BAR],
            color=line_colour(lv, INK),
            lw=4,
            solid_capstyle="butt",
        )

    assets = _assets_by_bus(data) if show_assets else {}
    for t in lay.terminals:
        tx = lay.x[t]
        ax.plot([tx, tx], [Y_BAR, Y_NODE], color=AXIS, lw=1, zorder=1)
        size = 28 + 0.55 * mean_kw.get(t, 0.0)
        if colour:
            face = colour(t)
        elif metered_buses is not None:
            face = BLUE if t in metered_buses else GREY
        else:
            face = BLUE
        ax.scatter(
            [tx], [Y_NODE], s=size, facecolor=face, edgecolor=INK2, lw=0.6, zorder=3
        )
        ax.text(
            tx,
            Y_LABEL,
            labels.get(t, t.split("_", 1)[-1]),
            rotation=90,
            ha="center",
            va="top",
            fontsize=label_size,
            color=INK2,
        )
        for k, kind in enumerate(assets.get(t, [])):
            ax.scatter(
                [tx + 0.3],
                [Y_BAR - 0.5 - 0.42 * k],
                s=34,
                linewidth=1.2,
                zorder=4,
                **_GLYPH[kind],
            )
    xs = [lay.x[t] for t in lay.terminals]
    ax.set_xlim(min(min(xs), lay.mv[0]) - 0.8, max(max(xs), lay.mv[1]) + 1.2)
    ax.set_ylim(-8.4, 1.3)
    ax.axis("off")
    return trafo_of


def _asset_legend(ax):
    items = [
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            markerfacecolor=BLUE,
            markeredgecolor=INK2,
            label="measured building (area = mean load)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            markerfacecolor=GREY,
            markeredgecolor=INK2,
            label="unmetered load (estimated from E.ON)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            ls="",
            markerfacecolor=ORANGE,
            markeredgecolor=ORANGE,
            label="existing PV (9 kWp)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            ls="",
            markerfacecolor="none",
            markeredgecolor=ORANGE,
            label="candidate PV roof",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            ls="",
            markerfacecolor="none",
            markeredgecolor=AQUA,
            label="EV charger site (assumed)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            ls="",
            markerfacecolor=BLUE,
            markeredgecolor=BLUE,
            label="NaS battery",
        ),
        Line2D(
            [0],
            [0],
            marker="h",
            ls="",
            markerfacecolor="none",
            markeredgecolor=VIOLET,
            label="heat pump (no own data)",
        ),
    ]
    ax.legend(
        handles=items,
        loc="lower left",
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.0, -0.02),
    )


def network_topology(
    data: CampusData, csv: pd.DataFrame, metered_loads: set, out: Path
) -> Path:
    lay = build_layout(data)
    mean_kw = mean_kw_by_bus(data, csv)
    metered_buses = set(data.loads.loc[data.loads["id"].isin(metered_loads), "bus"])
    fig, ax = plt.subplots(figsize=(17, 6.2))
    trafo_of = draw_network(ax, lay, data, mean_kw, metered_buses=metered_buses)
    # per-transformer mean load, printed under the transformer name
    per = {}
    for bus, kw in mean_kw.items():
        per[trafo_of.get(bus)] = per.get(trafo_of.get(bus), 0.0) + kw
    for tid, tx in lay.trafo_x.items():
        short = tid.replace("TR_", "")
        ax.text(
            tx + 0.5,
            Y_TRAFO - 0.62,
            f"avg {per.get(tid, 0) / 1000:.2f} MW",
            fontsize=6.5,
            color=MUTED,
            va="center",
        )
        ax.text(
            lay.bars[lay.trafo_lv[tid]][0],
            Y_BAR + 0.28,
            f"LV bus {short}",
            fontsize=6.5,
            color=MUTED,
        )
    _asset_legend(ax)
    fig.suptitle(
        "KFKI campus network: structure, loads and assets",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def network_voltage_maps(
    data: CampusData,
    csv: pd.DataFrame,
    details: Path,
    panels: list[tuple[str, str, str]],
    out: Path,
) -> Path | None:
    """Rows of the schematic coloured by max or min bus voltage for given scenarios.

    ``panels`` = [(scenario id, 'max'|'min', title), ...].
    """
    panels = [p for p in panels if (Path(details) / f"{p[0]}.npz").exists()]
    if not panels:
        return None
    lay = build_layout(data)
    mean_kw = mean_kw_by_bus(data, csv)
    norm = TwoSlopeNorm(vmin=0.92, vcenter=1.0, vmax=1.06)
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(17, 4.6 * len(panels)), squeeze=False
    )
    for ax, (sid, how, title) in zip(axes[:, 0], panels):
        z = np.load(Path(details) / f"{sid}.npz")
        fn = np.nanmax if how == "max" else np.nanmin
        vals = dict(zip(map(str, z["bus_names"]), fn(z["vm_pu"], axis=0)))
        draw_network(
            ax,
            lay,
            data,
            mean_kw,
            values=vals,
            norm=norm,
            show_assets=True,
            label_size=6.0,
        )
        ax.text(
            lay.mv[0] - 0.4,
            0.9,
            f"{title}\n({'highest' if how == 'max' else 'lowest'} voltage per bus, pu)",
            fontsize=9,
            va="top",
            color=INK,
        )
        worst = max(vals, key=lambda b: abs(vals[b] - 1.0))
        ax.text(
            lay.mv[0] - 0.4,
            -8.4,
            f"extreme: {vals[worst]:.3f} pu at {pretty_bus(worst, _labels(data))}",
            fontsize=8,
            color=INK2,
        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_V)
    cb = fig.colorbar(sm, ax=axes[:, 0].tolist(), shrink=0.6, pad=0.01)
    cb.set_label("bus voltage (pu)", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out
