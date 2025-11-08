import json
from collections import defaultdict
from os.path import join

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from utils.configuration import Config


def elegant_draw_network(
    net,
    ax=None,
    figsize=(10, 8),
    annotate=True,
    annotate_label="name",
    label_fontsize=11,
    bus_marker_size=300,
    load_size=120,
    sgen_size=120,
    legend_loc="lower right",
    output_path=None,
    show=True,
):
    # Try to extract geo coordinates into x/y if geo column exists
    if "geo" in net.bus.columns and (
        "x" not in net.bus.columns or "y" not in net.bus.columns
    ):

        def _extract_geo(g):
            if pd.isna(g):
                return pd.Series([np.nan, np.nan])
            try:
                obj = json.loads(g)
                coords = obj.get("coordinates")
                if coords and len(coords) >= 2:
                    return pd.Series([coords[0], coords[1]])
            except Exception:
                return pd.Series([np.nan, np.nan])

        xy = net.bus["geo"].apply(_extract_geo)
        xy.columns = ["x", "y"]
        net.bus[["x", "y"]] = xy

    # Build bus_coords mapping only for canonical bus indices (net.bus.index)
    bus_coords = {}
    for idx, row in net.bus.iterrows():
        x = row.get("x") if "x" in net.bus.columns else None
        y = row.get("y") if "y" in net.bus.columns else None
        if x is None or y is None or pd.isna(x) or pd.isna(y):
            bus_coords[idx] = None
        else:
            bus_coords[idx] = (float(x), float(y))
            bus_coords[str(idx)] = bus_coords[idx]

    # build name->index map so elements referencing bus names are resolved
    name_to_idx = {}
    for idx in net.bus.index:
        if idx not in net.bus.index:
            continue
        nm = net.bus.at[idx, "name"]
        if pd.notna(nm):
            name_to_idx[str(nm)] = idx

    # Build topology graph
    G = nx.Graph()
    for b in net.bus.index:
        G.add_node(b)

    for _, r in net.line.iterrows():
        G.add_edge(r.from_bus, r.to_bus)

    for _, r in net.trafo.iterrows():
        G.add_edge(r.hv_bus, r.lv_bus)

    # Helper to lookup a position for various key types
    def _pos(key):
        if key is None:
            return None
        # direct
        if key in bus_coords and bus_coords[key] is not None:
            return bus_coords[key]
        # if key matches a bus name
        ks = str(key)
        if ks in name_to_idx:
            return bus_coords.get(name_to_idx[ks])
        # try integer conversion
        k2 = int(key)
        if k2 in bus_coords and bus_coords[k2] is not None:
            return bus_coords[k2]
        k3 = str(key)
        if k3 in bus_coords and bus_coords[k3] is not None:
            return bus_coords[k3]
        return None

    # Prepare axis
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()
    ax.cla()

    for _, r in net.line.iterrows():
        p1 = _pos(r.from_bus)
        p2 = _pos(r.to_bus)
        if p1 is None or p2 is None:
            continue
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="k", lw=1.6, zorder=1)

    # Draw trafos
    for _, r in net.trafo.iterrows():
        p1 = _pos(r.hv_bus)
        p2 = _pos(r.lv_bus)
        if p1 is None or p2 is None:
            continue
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="#444444",
            lw=2.0,
            linestyle=(0, (3, 2)),
            zorder=2,
        )

    # Draw ext_grid
    for _, r in net.ext_grid.iterrows():
        p = _pos(r.bus)
        if p is None:
            continue
        ax.scatter(
            p[0], p[1], marker="*", color="gold", s=220, zorder=6, edgecolors="k"
        )

    # Compute spans for offsets
    xs_vals = compute_spans(bus_coords, net, 0)
    ys_vals = compute_spans(bus_coords, net, 1)
    if xs_vals and ys_vals:
        x_span = max(xs_vals) - min(xs_vals)
        y_span = max(ys_vals) - min(ys_vals)
        avg_span = max(x_span, y_span, 1.0)
    else:
        avg_span = 100.0

    # Draw buses and labels (iterate canonical bus indices)
    xs = []
    ys = []
    for idx in net.bus.index:
        pos = bus_coords.get(idx)
        if pos is None:
            continue
        xs.append(pos[0])
        ys.append(pos[1])

    ax.scatter(
        xs, ys, s=bus_marker_size, facecolor="#f0f0f0", edgecolor="#444444", zorder=5
    )

    if annotate:
        # increase label offset a bit for visibility
        label_offset = 0.00008 * avg_span
        for idx in net.bus.index:
            pos = bus_coords.get(idx)
            if pos is None:
                continue
            x, y = pos
            if annotate_label == "name" and "name" in net.bus.columns:
                lab = str(net.bus.at[idx, "name"])
            elif annotate_label == "index":
                lab = str(idx)
            elif annotate_label in net.bus.columns:
                lab = str(net.bus.at[idx, annotate_label])
            else:
                lab = str(idx)
            ax.text(
                x + label_offset,
                y + label_offset,
                lab.replace("LV1.101 ", ""),
                fontsize=label_fontsize,
                ha="left",
                va="bottom",
                zorder=12,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.2),
            )

    # Collect elements per bus and draw them around the bus so they don't overlap
    elems = defaultdict(list)

    # Normalize referenced bus key to canonical net.bus.index value (or None)
    def normalize_key(key):
        if key is None:
            return None
        # direct canonical
        if key in net.bus.index:
            return key
        # if key is a bus name
        ks = str(key)
        if ks in name_to_idx:
            return name_to_idx[ks]
        # try int conversion
        k2 = int(key)
        if k2 in net.bus.index:
            return k2
        # try string conversion
        k3 = str(key)
        if k3 in net.bus.index:
            return k3
        # fallback: try to find any variant present in bus_coords that maps back to a canonical index
        for cand in (key, str(key)):
            if cand in bus_coords:
                coord = bus_coords[cand]
                if coord is None:
                    continue
                for idx in net.bus.index:
                    if bus_coords.get(idx) == coord:
                        return idx
        return None

    # sgens (PV)
    for _, r in net.sgen.iterrows():
        k = normalize_key(r.bus)
        if k is not None:
            if "battery" in r["name"]:
                elems[k].append(("storage", r))
            else:
                elems[k].append(("sgen", r))
    # loads
    for _, r in net.load.iterrows():
        k = normalize_key(r.bus)
        if k is not None:
            elems[k].append(("load", r))

    # switches
    for _, r in net.switch.iterrows():
        k = normalize_key(r.bus)
        if k is not None:
            elems[k].append(("switch", r))

    # draw elements around bus centers
    angle_spread = np.deg2rad(18)
    for b, items in elems.items():
        p = _pos(b)
        if p is None:
            continue
        x0, y0 = p
        # order by preferred type so positions are stable
        type_order = {"sgen": 0, "load": 1, "storage": 3, "switch": 4}
        items_sorted = sorted(items, key=lambda it: type_order.get(it[0], 99))
        counts = defaultdict(int)
        for typ, _ in items_sorted:
            counts[typ] += 1
        r = max(0.0001 * avg_span, 0.0000025)
        base_angles = {
            "sgen": np.pi / 2.0,
            "load": -np.pi / 2.0,
            "storage": np.pi,
            "switch": np.pi / 4.0,
        }
        drawn = defaultdict(int)
        for typ, robj in items_sorted:
            idx = drawn[typ]
            ntyp = counts[typ]
            drawn[typ] += 1
            base = base_angles.get(typ, 0.0)
            if ntyp == 1:
                angle = base
            else:
                angle = base + (idx - (ntyp - 1) / 2.0) * angle_spread
            x_plot = x0 + r * np.cos(angle)
            y_plot = y0 + r * np.sin(angle)
            # x_plot = x0
            # y_plot = y0
            if typ == "sgen":
                ax.scatter(
                    x_plot,
                    y_plot,
                    marker="o",
                    s=sgen_size,
                    facecolor="#88ccee",
                    edgecolor="#2b6f8f",
                    zorder=14,
                )
                dx = 0.00005 * avg_span
                ax.plot(
                    [x_plot - dx, x_plot + dx],
                    [y_plot, y_plot],
                    color="#2b6f8f",
                    lw=1.2,
                    zorder=15,
                )
            elif typ == "load":
                ax.scatter(
                    x_plot,
                    y_plot,
                    marker="v",
                    s=load_size,
                    facecolor="#ffffff",
                    edgecolor="#cc4444",
                    zorder=14,
                )
            elif typ == "storage":
                ax.scatter(
                    x_plot,
                    y_plot,
                    marker="P",
                    s=120,
                    facecolor="#b3e6b3",
                    edgecolor="#2d8f2d",
                    zorder=13,
                )

    # legend (unchanged)
    pv_circle = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#88ccee",
        markeredgecolor="#2b6f8f",
        markersize=8,
        linestyle="None",
    )
    pv_hline = Line2D([-0.2, 0.2], [0, 0], color="#2b6f8f", linewidth=1)
    pv_proxy = (pv_circle, pv_hline)
    load_proxy = Line2D(
        [0],
        [0],
        marker="v",
        color="#cc4444",
        markerfacecolor="#ffffff",
        markeredgecolor="#cc4444",
        markersize=8,
        linestyle="None",
    )
    bus_proxy = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#f0f0f0",
        markeredgecolor="#444444",
        markersize=8,
        linestyle="None",
    )
    extgrid_proxy = Line2D(
        [0],
        [0],
        marker="*",
        color="gold",
        markerfacecolor="gold",
        markersize=10,
        linestyle="None",
    )
    stor_proxy = Line2D(
        [0],
        [0],
        marker="P",
        color="w",
        markerfacecolor="#b3e6b3",
        markeredgecolor="#2d8f2d",
        markersize=8,
        linestyle="None",
    )

    legend_handles = [pv_proxy, load_proxy, bus_proxy, extgrid_proxy, stor_proxy]
    legend_labels = ["PV", "Load", "Bus", "External grid", "Storage"]
    ax.legend(
        legend_handles,
        legend_labels,
        loc=legend_loc,
        framealpha=0.95,
        fontsize="small",
        handler_map={tuple: HandlerTuple()},
    )

    # finalize
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    xs_all = compute_spans(bus_coords, net, 0)
    ys_all = compute_spans(bus_coords, net, 1)
    if xs_all and ys_all:
        xmin, xmax = min(xs_all), max(xs_all)
        ymin, ymax = min(ys_all), max(ys_all)
        xpad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
        ypad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

    plt.tight_layout()
    if output_path:
        plt.savefig(join(output_path, "network.png"), dpi=300)
    if show:
        plt.show()

    return fig, ax


def compute_spans(bus_coords, net, idx):
    xs_vals = [
        p[idx] for k, p in bus_coords.items() if p is not None and (k in net.bus.index)
    ]
    return xs_vals


if __name__ == "__main__":
    import simbench

    net = simbench.get_simbench_net("1-LV-rural1--0-sw")
    elegant_draw_network(net, output_path=Config().get("path", "output"))
