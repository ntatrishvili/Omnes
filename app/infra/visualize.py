import json
from collections import defaultdict
from copy import copy
from os.path import join
from typing import List, Union, Tuple, Optional

import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
import pulp
from matplotlib import pyplot as plt, dates as mdates, cm as cm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path
from pandas import Timestamp, Timedelta

from app.infra.configuration import Config

mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["figure.facecolor"] = "none"
mpl.rcParams["axes.facecolor"] = "none"

# Define a named, harmonious palette for the project (modern, vibrant, scientific)
OMNES_PALETTE = {
    "fluorescent_green": "#39FF14",  # vivid green (accent)
    "fluorescent_pink": "#FF2D95",  # vivid pink (accent)
    "deep_teal": "#0B6E66",
    "soft_cyan": "#88CDEE",
    "deep_green": "#2D8F2D",
    "soft_green": "#B3E6B3",
    "coral_red": "#CC4444",
    "gold": "#FFD700",
    "neutral_light": "#F0F0F0",
    "light_gray": "#D3D3D3",
    "dark_gray": "#444444",
    "navy": "#013A63",
    "magenta": "#D61C6F",
    "purple": "#6B2D9A",
    "black": "#000000",
    "white": "#FFFFFF",
}

# Set a default matplotlib color cycle using key palette colors
_default_cycle = [
    OMNES_PALETTE["fluorescent_green"],
    OMNES_PALETTE["fluorescent_pink"],
    OMNES_PALETTE["deep_teal"],
    OMNES_PALETTE["soft_cyan"],
    OMNES_PALETTE["coral_red"],
    OMNES_PALETTE["soft_green"],
    OMNES_PALETTE["navy"],
]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=_default_cycle)


def elegant_draw_network(
    net,
    ax=None,
    figsize=(10, 8),
    annotate=True,
    annotate_label="name",
    label_fontsize=18,
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
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=OMNES_PALETTE["dark_gray"],
            lw=1.6,
            zorder=1,
        )

    # Draw trafos
    for _, r in net.trafo.iterrows():
        p1 = _pos(r.hv_bus)
        p2 = _pos(r.lv_bus)
        if p1 is None or p2 is None:
            continue
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=OMNES_PALETTE["dark_gray"],
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
            p[0],
            p[1],
            marker="*",
            color=OMNES_PALETTE["gold"],
            s=220,
            zorder=6,
            edgecolors=OMNES_PALETTE["dark_gray"],
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
        xs,
        ys,
        s=bus_marker_size,
        facecolor=OMNES_PALETTE["neutral_light"],
        edgecolor=OMNES_PALETTE["dark_gray"],
        zorder=5,
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
                bbox=dict(
                    facecolor=OMNES_PALETTE["white"],
                    alpha=0.85,
                    edgecolor="none",
                    pad=0.2,
                ),
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
                    facecolor=OMNES_PALETTE["soft_cyan"],
                    edgecolor=OMNES_PALETTE["deep_teal"],
                    zorder=14,
                )
                dx = 0.00005 * avg_span
                ax.plot(
                    [x_plot - dx, x_plot + dx],
                    [y_plot, y_plot],
                    color=OMNES_PALETTE["deep_teal"],
                    lw=1.2,
                    zorder=15,
                )
            elif typ == "load":
                ax.scatter(
                    x_plot,
                    y_plot,
                    marker="v",
                    s=load_size,
                    facecolor=OMNES_PALETTE["white"],
                    edgecolor=OMNES_PALETTE["coral_red"],
                    zorder=14,
                )
            elif typ == "storage":
                ax.scatter(
                    x_plot,
                    y_plot,
                    marker="P",
                    s=120,
                    facecolor=OMNES_PALETTE["soft_green"],
                    edgecolor=OMNES_PALETTE["deep_green"],
                    zorder=13,
                )

    # legend (unchanged)
    pv_circle = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=OMNES_PALETTE["soft_cyan"],
        markeredgecolor=OMNES_PALETTE["deep_teal"],
        markersize=8,
        linestyle="None",
    )
    pv_hline = Line2D(
        [-0.2, 0.2], [0, 0], color=OMNES_PALETTE["deep_teal"], linewidth=1
    )
    pv_proxy = (pv_circle, pv_hline)
    load_proxy = Line2D(
        [0],
        [0],
        marker="v",
        color=OMNES_PALETTE["coral_red"],
        markerfacecolor=OMNES_PALETTE["white"],
        markeredgecolor=OMNES_PALETTE["coral_red"],
        markersize=8,
        linestyle="None",
    )
    bus_proxy = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=OMNES_PALETTE["neutral_light"],
        markeredgecolor=OMNES_PALETTE["dark_gray"],
        markersize=8,
        linestyle="None",
    )
    extgrid_proxy = Line2D(
        [0],
        [0],
        marker="*",
        color=OMNES_PALETTE["gold"],
        markerfacecolor=OMNES_PALETTE["gold"],
        markersize=10,
        linestyle="None",
    )
    stor_proxy = Line2D(
        [0],
        [0],
        marker="P",
        color="w",
        markerfacecolor=OMNES_PALETTE["soft_green"],
        markeredgecolor=OMNES_PALETTE["deep_green"],
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
        fontsize="large",
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


def plot_branch_voltage_heatmaps(
    results: List[Union[str, pd.DataFrame]],
    branch_buses: List,
    bus_names: List[str],
    scenario_names: List[str] = None,
    cmap: str = "Greys",
    figsize: Tuple[float, float] = (12, 3.5),
    savepath: str = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot stacked heatmaps (max, mean, min) of node voltages along a branch for multiple scenarios.

    Parameters
    - results: list of CSV file paths or pandas DataFrames. Each entry corresponds to one scenario.
               Each DataFrame must contain a column identifying the bus (e.g. 'bus','node','bus_name')
               and a voltage column (e.g. 'vm_pu','vm','voltage').
    - branch_buses: ordered list of bus names (x-axis).
    - scenario_names: optional list of names for the scenarios (y-axis). If None, indices are used.
    - cmap: matplotlib colormap name (defaults to 'Greys' to match network plotting).
    - figsize: figure size tuple.
    - savepath: if provided, save the figure to this path.

    Returns:
    - (fig, axes): Matplotlib figure and array of axes.
    """
    n_scenarios = len(results)
    n_buses = len(branch_buses)

    # arrays to store per-scenario statistics
    max_arr = np.full((n_scenarios, n_buses), np.nan, dtype=float)
    max2_arr = np.full((n_scenarios, n_buses), np.nan, dtype=float)
    mean_arr = np.full((n_scenarios, n_buses), np.nan, dtype=float)
    min_arr = np.full((n_scenarios, n_buses), np.nan, dtype=float)

    for i, res in enumerate(results):
        # load DataFrame
        if isinstance(res, str):
            df = pd.read_csv(res)
        elif isinstance(res, pd.DataFrame):
            df = res.copy()
        else:
            raise ValueError(
                "Each item of results must be a filepath or a pandas DataFrame"
            )

        # filter branch buses and aggregate
        branch_bus_indices = (
            net.bus[net.bus.name.str.split(" ").str[-1].astype(int).isin(branch_buses)]
            .index.astype(str)
            .tolist()
        )
        df_branch = df[
            [
                c
                for c in df.columns
                if "vm_pu" in c and c.split("_")[0] in branch_bus_indices
            ]
        ]
        if df_branch.empty:
            # leave NaNs if nothing found
            continue

        # build series for each bus in requested order
        for j, bus in enumerate(branch_bus_indices):
            try:
                s = df_branch[f"{bus}_vm_pu"]
            except KeyError:
                continue
            max2_arr[i, j] = s.max(skipna=True)
            max_arr[i, j] = s.quantile(0.95)
            min_arr[i, j] = s.std(skipna=True)
            mean_arr[i, j] = s.median(skipna=True)

    # prepare scenario names
    if scenario_names is None:
        scenario_names = [f"S{i + 1}" for i in range(n_scenarios)]
    if len(scenario_names) != n_scenarios:
        raise ValueError("Length of scenario_names must match number of results")

    # plotting
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True)
    titles = ["Maximum", "95% quantile", "Median", "Std. deviation"]
    data_list = [max2_arr, max_arr, mean_arr, min_arr]
    # fallback if all NaN
    for ax, data, title in zip(axes, data_list, titles):
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        if np.isnan(vmin) or np.isnan(vmax):
            vmin, vmax = 0.0, 1.0
        # im = ax.imshow(data, cmap=cmap, aspect="equal", origin="lower", vmin=vmin, vmax=vmax)
        im = ax.imshow(
            data,
            cmap=cmap,
            aspect="equal",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        im.set_zorder(1)

        # --- grid between heatmap cells ---
        # minor ticks at cell boundaries
        ax.set_xticks(np.arange(-0.5, n_buses, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_scenarios, 1), minor=True)
        # draw minor grid (cell separators)
        ax.grid(
            which="minor",
            color="lightgrey",
            linestyle="-",
            linewidth=0.6,
            zorder=2,
            alpha=0.9,
        )
        # keep major ticks for labels, disable their grid lines
        ax.grid(which="major", visible=False)
        # ensure ticks align with cells
        ax.set_xlim(-0.5, n_buses - 0.5)
        ax.set_ylim(-0.5, n_scenarios - 0.5)

        ax.set_title(title)
        ax.set_ylabel("Scenarios")
        # y ticks: origin='lower' -> scenario 0 at bottom (matches sample where N1 is bottom)
        ax.set_yticks(np.arange(n_scenarios))
        ax.set_yticklabels(scenario_names)
        # add colorbar on the right of each subplot
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.06, fraction=0.05)
        # nice formatting for colorbar label (optional)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_ylabel("Voltage [pu]")

        # x ticks for busbars on bottom subplot only
        ax.set_xticks(np.arange(n_buses))
        ax.set_xticklabels(
            [f"{name}" for bus, name in zip(branch_buses, bus_names)], rotation=90
        )
        ax.set_xlabel("Buses")

    plt.suptitle("Node voltages")
    plt.tight_layout()
    if savepath:
        plt.savefig(join(savepath, "branch_voltages.png"), dpi=400, bbox_inches="tight")
    plt.show()
    return fig, axes


def plot_losses_violations_heatmaps(
    results: list,
    net,
    scenario_names: list = None,
    cmap: str = "Greys",
    figsize: tuple = (10, 6),
    savepath: str = None,
    violation_threshold: float = 1.01,
) -> tuple:
    """
    Plot two heatmaps (losses, voltage violations) across scenarios.
    - results: list of CSV paths or pandas DataFrames (each scenario)
    - net: pandapower network (used to get line r and length)
    - returns (fig, axes)
    """

    n_scenarios = len(results)
    stats_cols = ["Maximum", "95% quantile", "Median", "Std. deviation"]

    # increase overall fontsize by 4 points for this plot
    font_increase = 4
    base_font = mpl.rcParams.get("font.size", 10)
    fs = base_font + font_increase

    # precompute per-line r * length (ohm)
    line_rlen = {}
    for idx, row in net.line.iterrows():
        r = row.get("r_ohm_per_km", 0.0) or 0.0
        length = row.get("length_km", 0.0) or 0.0
        line_rlen[int(idx)] = float(r) * float(length)  # ohm (km cancels)

    # prepare arrays
    losses_arr = np.full((n_scenarios, len(stats_cols)), np.nan, dtype=float)
    viols_arr = np.full((n_scenarios, len(stats_cols)), np.nan, dtype=float)

    for i, res in enumerate(results):
        # load dataframe
        if isinstance(res, str):
            df = pd.read_csv(res, index_col=None)
        elif isinstance(res, pd.DataFrame):
            df = res.copy()
        else:
            raise ValueError(
                "Each item of results must be a filepath or a pandas DataFrame"
            )

        # identify columns
        i_cols = [c for c in df.columns if c.endswith("_i_ka")]
        vm_cols = [c for c in df.columns if c.endswith("_vm_pu")]

        if df.empty or (not i_cols and not vm_cols):
            continue

        # compute per-timestep total losses (W) and per-timestep violation counts
        total_losses = []

        # For violations we will compute number of hours per day where ANY bus exceeds
        # the violation_threshold. This yields integer counts per day (0..24).
        # Create a datetime index for grouping into days if not already datetime-like.
        try:
            is_datetime = pd.api.types.is_datetime64_any_dtype(df.index)
        except Exception:
            is_datetime = False
        if not is_datetime:
            parsed_index = pd.date_range(start="2016-01-01", periods=len(df), freq="1h")
            df_indexed = df.copy()
            df_indexed.index = parsed_index
        else:
            df_indexed = df.copy()

        # compute per-timestep total losses and also prepare vm dataframe for violations
        vm_df = df_indexed[vm_cols].astype(float) if vm_cols else pd.DataFrame()

        for _, row in df_indexed.iterrows():
            # losses
            loss_sum_W = 0.0
            for c in i_cols:
                line_idices = c.split("_")
                if len(line_idices) < 1:
                    continue
                line_index = line_idices[0]
                val = row.get(c, np.nan)
                if pd.isna(val):
                    continue
                I_A = float(val) * 1000.0  # kA -> A
                r_len = line_rlen.get(line_index, 0.0)
                # approximate three-phase losses: 3 * I^2 * R_total (W)
                loss_sum_W += 3.0 * (I_A**2) * r_len
            total_losses.append(loss_sum_W)

        # compute daily violation counts: for each day count hours where any vm > threshold
        if not vm_df.empty:
            # boolean DataFrame where True indicates a violation at that timestep for that bus
            violation_bool = vm_df > violation_threshold
            # any bus violation per timestep
            any_violation_per_t = violation_bool.any(axis=1)
            # group by day (date) using Grouper and count True values per day -> integer counts
            # This is robust for datetime-like indices and avoids static-analysis warnings.
            daily_counts = (
                any_violation_per_t.groupby(pd.Grouper(freq="D"))
                .sum()
                .values.astype(float)
            )
        else:
            daily_counts = np.array([], dtype=float)

        # compute statistics (max, 95th, median, std) across days for violations
        arr_losses = np.array(total_losses, dtype=float)

        if arr_losses.size > 0 and not np.all(np.isnan(arr_losses)):
            losses_arr[i, 0] = np.nanmax(arr_losses)
            losses_arr[i, 1] = np.nanpercentile(arr_losses, 95)
            losses_arr[i, 2] = np.nanmedian(arr_losses)
            losses_arr[i, 3] = np.nanstd(arr_losses)

        if daily_counts.size > 0 and not np.all(np.isnan(daily_counts)):
            # statistics across days -> round to nearest integers as requested
            max_d = np.nanmax(daily_counts)
            p95_d = np.nanpercentile(daily_counts, 95)
            med_d = np.nanmedian(daily_counts)
            std_d = np.nanstd(daily_counts)
            # round to nearest integer
            viols_arr[i, 0] = int(np.rint(max_d))
            viols_arr[i, 1] = int(np.rint(p95_d))
            viols_arr[i, 2] = int(np.rint(med_d))
            viols_arr[i, 3] = int(np.rint(std_d))

    # scenario names
    if scenario_names is None:
        scenario_names = [f"S{i + 1}" for i in range(n_scenarios)]
    if len(scenario_names) != n_scenarios:
        raise ValueError("Length of scenario_names must match number of results")

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    titles = [
        "Total line losses [W]",
        "Count of U(pu) > {:.2f} per day".format(violation_threshold),
    ]
    data_list = [losses_arr, viols_arr]

    for ax, data, title in zip(axes, data_list, titles):
        # handle empty/all-nan
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        if np.isnan(vmin) or np.isnan(vmax):
            vmin, vmax = 0.0, 1.0

        # For the violations heatmap use discrete integer colors
        if np.array_equal(data, viols_arr):
            # ensure integer bounds
            vi_min = int(np.nanmin(data))
            vi_max = int(np.nanmax(data))
            if np.isnan(vi_min) or np.isnan(vi_max):
                vi_min, vi_max = 0, 1
            if vi_max < vi_min:
                vi_max = vi_min
            n_colors = max(1, vi_max - vi_min + 1)
            cmap_obj = plt.get_cmap(cmap, n_colors)
            im = ax.imshow(
                data,
                cmap=cmap_obj,
                aspect="equal",
                origin="lower",
                vmin=vi_min,
                vmax=vi_max,
                interpolation="nearest",
            )
        else:
            im = ax.imshow(
                data,
                cmap=cmap,
                aspect="equal",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
        im.set_zorder(1)

        # grid lines
        n_cols = data.shape[1]
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_scenarios, 1), minor=True)
        ax.grid(
            which="minor",
            color="lightgrey",
            linestyle="-",
            linewidth=0.6,
            zorder=2,
            alpha=0.9,
        )
        ax.grid(which="major", visible=False)
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(-0.5, n_scenarios - 0.5)

        ax.set_title(title, fontsize=fs)
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(stats_cols, rotation=45, ha="right", fontsize=fs)
        ax.set_yticks(np.arange(n_scenarios))
        ax.set_yticklabels(scenario_names, fontsize=fs)
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.06, fraction=0.06)
        cbar.ax.tick_params(labelsize=fs)
        # also set colorbar label size if present
        cbar.ax.yaxis.label.set_size(fs)

        # If this is the violations heatmap, show only integer ticks 0,1,2,... up to max
        try:
            is_viol = np.array_equal(data, viols_arr)
        except Exception:
            is_viol = False

        if is_viol:
            valid = data[~np.isnan(data)]
            if valid.size > 0:
                vi_max = int(np.nanmax(valid))
                if vi_max < 0:
                    vi_max = 0
            else:
                vi_max = 0
            ticks = np.arange(0, vi_max + 1)
            # set ticks and integer labels
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([str(int(t)) for t in ticks])
            cbar.ax.tick_params(labelsize=fs)
            # adjust color limits to align with integer bins
            if len(ticks) > 0:
                cbar.mappable.set_clim(ticks[0] - 0.5, ticks[-1] + 0.5)

    plt.suptitle(
        "Network losses and voltage violations across scenarios", fontsize=fs + 2
    )
    plt.tight_layout()
    if savepath:
        plt.savefig(
            join(savepath, "losses_violations_heatmaps.png"),
            dpi=400,
            bbox_inches="tight",
        )
    plt.show()
    return fig, axes


def fit_network_axis(
    ax, net, padx=0.05, pady=0.05, min_span=1e-6, ymargin=0, xmargin=0
):
    """
    Tighten `ax` limits to the extents of net.bus['x','y'] with a fractional padding.
    - pad: fraction of span to add on each side (0.05 = 5%)
    - min_span: avoid zero-span if all coords identical
    """
    if "x" not in net.bus.columns or "y" not in net.bus.columns:
        return
    coords = net.bus[["x", "y"]].dropna()
    if coords.empty:
        return
    x_min, x_max = coords["x"].min(), coords["x"].max()
    y_min, y_max = coords["y"].min(), coords["y"].max()
    x_span = max(x_max - x_min, min_span)
    y_span = max(y_max - y_min, min_span)
    pad_x = x_span * padx
    pad_y = y_span * pady
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale(enable=False)
    ax.margins(xmargin, ymargin)


def annotate_buses(
    ax, net, label="name", fontsize=16, offset=(0.0, 0.0), bbox=True, filter_fn=None
):
    """
    Annotate buses on a matplotlib axis created by pandapower.simple_plot.

    Parameters
    - ax: matplotlib Axes where the network was drawn
    - net: pandapower network
    - label: "name", "index" or any column name from net.bus (e.g. "vm_pu")
    - fontsize: text size
    - offset: (dx, dy) offset in data coordinates to move labels from bus position
    - bbox: whether to draw a semi-transparent background box
    - filter_fn: optional function (idx, row) -> bool to skip some buses
    """
    for idx, row in net.bus.iterrows():
        if filter_fn and not filter_fn(idx, row):
            continue
        x, y = row.get("x"), row.get("y")
        if pd.isna(x) or pd.isna(y):
            continue
        if label == "name":
            txt = (
                row.get("name", str(idx))
                .replace("LV1.101 Bus ", "")
                .replace("MV1.101 Bus 4", "MV grid")
            )
        elif label == "index":
            txt = str(idx)
        else:
            txt = str(row.get(label, ""))
        bbox_kw = (
            dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.2)
            if bbox
            else None
        )
        ax.text(
            x + offset[0],
            y + offset[1],
            txt,
            fontsize=fontsize,
            ha="center",
            va="center",
            zorder=20,
            bbox=bbox_kw,
        )


def draw_battery_icon(ax, x, y, size=0.02, color="blue"):
    """
    Draw a stylized battery centered at (x, y).
    - size: fractional size relative to axis span (0.02 = 2% of axis span)
    """
    # ensure valid coordinates
    if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
        return

    # compute data-space size from axis spans
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_span = max(abs(x1 - x0), 1e-9)
    y_span = max(abs(y1 - y0), 1e-9)

    w = size * x_span
    h = size * 0.6 * y_span

    # small rectangle with a terminal
    verts = [
        (x - w / 2, y - h / 2),
        (x + w / 2, y - h / 2),
        (x + w / 2, y + h / 2),
        (x + w * 0.6, y + h / 2),
        (x + w * 0.6, y + h * 0.7),
        (x + w * 0.4, y + h * 0.7),
        (x + w * 0.4, y + h / 2),
        (x - w / 2, y + h / 2),
        (x - w / 2, y - h / 2),
    ]
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = PathPatch(
        path, facecolor=color, edgecolor="black", lw=1.0, zorder=30, alpha=0.9
    )
    ax.add_patch(patch)
    # small inner terminal line for contrast
    ax.plot(
        [x + w * 0.45, x + w * 0.6],
        [y + h * 0.5, y + h * 0.6],
        color="black",
        lw=1.0,
        zorder=31,
    )


def visualize_high_voltage_day(
    net,
    results_csv: Optional[str] = None,
    results_df: Optional[pd.DataFrame] = None,
    branches: Optional[List[List[str]]] = None,
    figsize=(16, 10),
    cmap="RdYlBu_r",
    base_year: int = 2016,
    output_path: Optional[str] = None,
    scenario: Optional[str] = None,
    trafo=4,
    remove_battery: bool = False,
):
    """Visualize network snapshot at the highest voltage timestep."""

    # --- Load and preprocess results ---
    if results_df is None:
        if results_csv is None:
            raise ValueError("Provide results_df or results_csv")
        results_df = pd.read_csv(results_csv, index_col=0)

    df = results_df.copy()

    # Extract vm_pu and i_ka columns
    vm_cols = [c for c in df.columns if c.endswith("_vm_pu")]
    i_cols = [c for c in df.columns if c.endswith("_i_ka")]
    if not vm_cols:
        raise ValueError("No voltage columns (_vm_pu) found")

    vm_df = df[vm_cols].astype(float)
    max_vm_per_t = vm_df.max(axis=1)
    peak_idx = max_vm_per_t.idxmax()
    peak_row = df.loc[peak_idx]

    # --- Extract coordinates before plotting ---
    def extract_coords(geo_str):
        if pd.isna(geo_str):
            return None, None
        try:
            geo = json.loads(geo_str)
            lon, lat = geo["coordinates"]
            return lon, lat
        except Exception:
            return None, None

    if "x" not in net.bus.columns or "y" not in net.bus.columns:
        net.bus[["x", "y"]] = net.bus["geo"].apply(
            lambda g: pd.Series(extract_coords(g))
        )

    # --- Peak timestep data ---
    peak_vm = {
        int(c.split("_")[0]): float(peak_row[c])
        for c in vm_cols
        if pd.notna(peak_row[c])
    }
    edge_currents = {
        int(c.split("_")[0]): float(peak_row[c])
        for c in i_cols
        if pd.notna(peak_row[c])
    }

    # --- Day subset around peak ---
    parsed_index = pd.date_range(start=f"{base_year}-01-01", periods=len(df), freq="1h")
    df.index = parsed_index
    peak_ts = df.iloc[peak_idx].name
    day_df = df.loc[peak_ts.date().strftime(f"{base_year}-%m-%d")]
    time_index = day_df.index

    # --- Prepare plot ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.5],
        hspace=0.24,
        wspace=0.24,
    )
    ax_net = fig.add_subplot(gs[0, 0])
    ax_branch = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[1, :])

    # --- Node colors based on vm_pu ---
    node_vals = np.array([peak_vm.get(i, np.nan) for i in net.bus.index])
    vmin, vmax = np.nanmin(node_vals), np.nanmax(node_vals)
    cmap_obj = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    node_colors = [
        cmap_obj(norm(v)) if np.isfinite(v) else "lightgray" for v in node_vals
    ]

    # --- Line colors and widths based on currents ---
    line_curr = np.array([edge_currents.get(i, np.nan) for i in net.line.index])
    cmin, cmax = np.nanmin(line_curr), np.nanmax(line_curr)
    edge_cmap = plt.get_cmap(cmap)
    line_colors = []
    line_widths = []
    for val in line_curr:
        if np.isnan(val):
            line_colors.append("lightgray")
            line_widths.append(0.5)
        else:
            frac = (val - cmin) / (cmax - cmin + 1e-9)
            line_colors.append(edge_cmap(frac))
            line_widths.append(1.0 + 4.0 * frac)

    # --- Draw network ---
    # --- Draw network with pandapower ---
    import pandapower.plotting as pplot

    # Compute color maps for bus voltages and line loadings
    net.res_bus = pd.DataFrame(
        {"vm_pu": [peak_vm.get(i, np.nan) for i in net.bus.index]}, index=net.bus.index
    )
    net.res_line = pd.DataFrame(
        {
            "loading_percent": [
                edge_currents.get(i, np.nan) * 1000 for i in net.line.index
            ]
        },
        index=net.line.index,
    )

    # Base plot
    pnet = copy(net)
    batteries = pnet.sgen[pnet.sgen["name"] == "battery"]
    if remove_battery:
        pnet.sgen.drop(batteries.index, inplace=True)
    pplot.simple_plot(
        pnet,
        ax=ax_net,
        bus_size=3.0,
        ext_grid_size=2,
        line_width=line_widths,
        show_plot=False,
        bus_color=node_colors,
        line_color=line_colors,
        plot_loads=True,
        plot_sgens=True,
        load_size=3.0,
        sgen_size=3.0,
    )

    fit_network_axis(ax_net, net, padx=0.08, pady=0.17)
    annotate_buses(ax_net, net, label="name", fontsize=16, offset=(0.00025, -0.0001))

    # --- Overlay elements (battery) ---
    battery_buses = net.bus.loc[batteries["bus"]]
    for i, row in batteries.iterrows():
        if not row.get("in_service", True):
            continue
        if row["bus"] not in net.bus:
            continue
        x, y = net.bus.loc[row["bus"], ["x", "y"]]
        draw_battery_icon(ax_net, x, y, size=0.07, color="blue")

    # --- Create a small legend inside the network axes using proxy artists ---
    # PV proxy: circle + horizontal line (tuple), handled by HandlerTuple
    pv_circle = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="white",
        markeredgecolor="k",
        markersize=8,
        linestyle="None",
    )
    pv_hline = Line2D([-0.2, 0.2], [0, 0], color="k", linewidth=1)

    pv_proxy = (pv_circle, pv_hline)

    # Load proxy: down-pointing triangle
    load_proxy = Line2D(
        [0],
        [0],
        marker="v",
        color="k",
        markerfacecolor="white",
        markeredgecolor="k",
        markersize=8,
        linestyle="None",
    )

    battery_proxy = Patch(facecolor="blue", edgecolor="k")
    extgrid_proxy = Line2D(
        [0],
        [0],
        marker="s",
        color="gold",
        markerfacecolor="gold",
        markersize=8,
        linestyle="None",
    )

    legend_handles = [pv_proxy, load_proxy, battery_proxy, extgrid_proxy]
    legend_labels = ["PV", "Load", "Battery", "External grid"]
    ax_net.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        framealpha=0.9,
        fontsize="medium",
    )

    ax_net.set_title(f"Node color=Voltage [pu], Edge color=Current [A]")
    ax_net.set_xlabel("Longitude")
    ax_net.set_ylabel("Latitude")

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    # Use make_axes_locatable to append small colorbar axes when possible.
    # However, in unit tests plt may be patched/mocked which makes axes
    # objects non-standard (MagicMock) and append_axes can fail with
    # unittest.mock.InvalidSpecError. Fall back gracefully to a simpler
    # fig.colorbar(..., ax=...) call if append_axes fails.
    try:
        divider = make_axes_locatable(ax_net)
        cax1 = divider.append_axes("right", size="3%", pad=0.09)
        cax2 = divider.append_axes("right", size="3%", pad=0.82)
        fig.colorbar(sm, cax=cax1, label="Voltage [pu]")
        cm = mpl.cm.ScalarMappable(
            cmap=edge_cmap, norm=mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        )
        cm.set_array([])
        fig.colorbar(cm, cax=cax2, label="Current [A]")
    except Exception:
        # Fallback: attach colorbars to the main axis; ignore failures silently
        try:
            fig.colorbar(sm, ax=ax_net, label="Voltage [pu]")
        except Exception:
            pass
        try:
            cm = mpl.cm.ScalarMappable(
                cmap=edge_cmap, norm=mpl.colors.Normalize(vmin=cmin, vmax=cmax)
            )
            cm.set_array([])
            fig.colorbar(cm, ax=ax_net, label="Current [A]")
        except Exception:
            pass

    # --- Branch voltage profiles ---
    highlight_colors = ["C1", "C2", "C3", "C4", "C5"]
    markers = ["o", "s", "D", "^", "v", "P", "*", "X"]
    bus_name_to_index = {row["name"]: idx for idx, row in net.bus.iterrows()}

    for i, branch in enumerate(branches or []):
        idxs = [
            bus_name_to_index[f"LV1.101 Bus {name}"]
            for name in branch
            if f"LV1.101 Bus {name}" in bus_name_to_index
        ]
        if idxs == []:
            continue
        vals = [peak_vm.get(idx, np.nan) for idx in idxs]
        m = markers[i % len(markers)]
        c = highlight_colors[i % len(highlight_colors)]
        ax_branch.plot(
            range(len(idxs)),
            vals,
            marker=m,
            markeredgecolor="darkgrey",
            markeredgewidth=0.5,
            markersize=10,
            color=c,
            label=f"Branch {i + 1}",
        )
        ax_branch.scatter(
            range(len(idxs))[-1],
            vals[-1],
            marker=m,
            edgecolor=c,
            s=250,
            c="white",
            linewidths=2,
            label=None,
        )
        bess_idx = [j for j, i in enumerate(idxs) if i in battery_buses.index]
        if bess_idx and scenario != "No battery":
            x_positions = bess_idx
            y_positions = [vals[j] for j in bess_idx]
            ax_branch.scatter(
                x_positions,
                y_positions,
                marker="s",
                edgecolor="black",
                s=200,
                c="blue",
                linewidths=2,
                label=None,
                zorder=10,
            )

        for j, name in enumerate(branch):
            ax_branch.text(
                j, vals[j] + 0.0001, name, fontsize=18, ha="center", va="bottom"
            )

    ax_branch.set_title("Voltage profile along branches")
    ax_branch.set_xlabel("Node order")
    ax_branch.set_ylabel("Voltage [pu]")
    # ax_branch.set_ylim(0.99, 1.02)
    ax_branch.legend(fontsize="medium")
    ax_branch.grid(True)

    # --- Day time-series (only branch endpoints) ---
    if branches:
        day_vm = day_df[[c for c in day_df.columns if c.endswith("_vm_pu")]].astype(
            float
        )
        for i, branch in enumerate(branches + [[trafo]]):
            last_name = branch[-1]
            last_idx = bus_name_to_index.get(f"LV1.101 Bus {last_name}")
            if last_idx is not None:
                col = f"{last_idx}_vm_pu"
                if col in day_vm.columns:
                    postfix = (
                        f"(branch {i + 1} end)"
                        if i != len(branches)
                        else "(transformer)"
                    )
                    ax_ts.plot(
                        time_index,
                        day_vm[col],
                        marker=markers[i % len(markers)],
                        markeredgecolor="darkgrey",
                        markeredgewidth=0.5,
                        markersize=7,
                        label=f"Bus {last_name} {postfix}",
                        color=highlight_colors[i % len(highlight_colors)],
                        lw=1.8,
                    )
    ax_ts.set_title("Voltage time series (day of peak timestep)")
    ax_ts.set_xlabel(f"Time [hours]\n{peak_ts:%Y-%m-%d}")
    ax_ts.set_ylabel("Voltage [pu]")
    # ax_ts.set_ylim(0.99, 1.02)
    ax_ts.grid(True)
    ax_ts.legend()

    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=45)
    title_text = f"Scenario: {scenario}" if scenario else "Scenario: default"
    fig.suptitle(
        f"Network snapshot at {peak_ts:%Y-%m-%d %H:00}\n{title_text}", fontsize=18
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(
            join(output_path, f"high_voltage_day_visualization{scenario}.png"), dpi=500
        )

    plt.show()


def plot_energy_flows(
    kwargs,
    pv_names,
    load_names,
    bess_names,
    slack_names,
    time_range_to_plot=None,
    output_path=None,
):
    time_set = kwargs["time_set"]
    nT = len(time_set)
    if time_range_to_plot is None:
        time_range_to_plot = range(nT)

    ax, twin_ax = None, None
    # --- Extract and sum all variables ---
    # PV production
    pv_profiles = {
        pv: np.array([pulp.value(kwargs[f"{pv}.p_out"][t]) for t in time_set])
        for pv in pv_names
    }
    pv_sum = sum(pv_profiles.values())
    print("PV total production:", pv_sum.max())

    # Loads
    load_profiles = {
        ld: np.array([pulp.value(kwargs[f"{ld}.p_cons"][t]) for t in time_set])
        for ld in load_names
    }
    load_sum = sum(load_profiles.values())
    print("Total load:", load_sum.max())

    # Battery flows
    bess_in_profiles = {
        b: np.array([pulp.value(kwargs[f"{b}.p_in"][t]) for t in time_set])
        for b in bess_names
    }
    bess_out_profiles = {
        b: np.array([pulp.value(kwargs[f"{b}.p_out"][t]) for t in time_set])
        for b in bess_names
    }
    e_bess_profiles = {
        b: np.array([pulp.value(kwargs[f"{b}.e_stor"][t]) for t in time_set])
        for b in bess_names
    }

    # Slack (import/export)
    slack_in_sum = np.sum(
        [
            np.array([pulp.value(kwargs[f"{s}.p_in"][t]) for t in time_set])
            for s in slack_names
        ],
        axis=0,
    )
    slack_out_sum = np.sum(
        [
            np.array([pulp.value(kwargs[f"{s}.p_out"][t]) for t in time_set])
            for s in slack_names
        ],
        axis=0,
    )

    # --- Compute basic metrics ---
    total_charge_hours = sum(np.sum(v > 0) for v in bess_in_profiles.values()) // 4
    total_discharge_hours = sum(np.sum(v > 0) for v in bess_out_profiles.values()) // 4
    print(f"Total charge hours: {total_charge_hours}")
    print(f"Total discharge hours: {total_discharge_hours}")

    # --- Plot setup ---
    plt.figure(figsize=(10, 6))
    colors_bess = cm.Greys(np.linspace(0.4, 0.8, len(bess_names))) if bess_names else []
    colors_pv = cm.Reds(np.linspace(0.5, 0.9, len(pv_names))) if pv_names else []
    colors_load = (
        cm.Greens(np.linspace(0.5, 0.9, len(load_names))) if load_names else []
    )

    # --- Plot PVs ---
    bottom_pv = np.zeros_like(time_set, dtype=float)
    for color, (pv, p) in zip(colors_pv, pv_profiles.items()):
        plt.bar(
            time_range_to_plot,
            -p[time_range_to_plot],
            color=color,
            width=1.0,
            label=f"{pv.replace('LV1.101 ', '')} (PV)",
            bottom=bottom_pv[time_range_to_plot],
        )
        bottom_pv[time_range_to_plot] -= p[time_range_to_plot]
    plt.plot(
        time_range_to_plot, -pv_sum[time_range_to_plot], "r", lw=2.5, label="Total PV"
    )

    # --- Plot loads ---
    bottom_loads = np.zeros_like(time_set, dtype=float)
    for color, (ld, p) in zip(colors_load, load_profiles.items()):
        plt.bar(
            time_range_to_plot,
            p[time_range_to_plot],
            color=color,
            width=1.0,
            label=f"{ld.replace('LV1.101 ', '')}",
            bottom=bottom_loads[time_range_to_plot],
        )
        bottom_loads[time_range_to_plot] += p[time_range_to_plot]

    plt.plot(
        time_range_to_plot,
        load_sum[time_range_to_plot],
        "--",
        lw=2.8,
        label="Total Load",
        color=colors_load[0],
    )

    # --- Plot battery charge/discharge ---
    bottom_charge = np.zeros_like(time_set, dtype=float)
    bottom_discharge = np.zeros_like(time_set, dtype=float)
    for color, (b, p_in) in zip(colors_bess, bess_in_profiles.items()):
        plt.bar(
            time_range_to_plot,
            p_in[time_range_to_plot],
            color=color,
            width=1.0,
            label=f"{b} charge",
            edgecolor="darkgrey",
            bottom=bottom_charge[time_range_to_plot],
        )
        bottom_charge[time_range_to_plot] += p_in[time_range_to_plot]
    for color, (b, p_out) in zip(colors_bess, bess_out_profiles.items()):
        plt.bar(
            time_range_to_plot,
            -p_out[time_range_to_plot],
            color=color,
            width=1.0,
            alpha=0.7,
            label=f"{b} discharge",
            edgecolor="darkgrey",
            bottom=bottom_discharge[time_range_to_plot],
        )
        bottom_discharge[time_range_to_plot] -= p_out[time_range_to_plot]

        # --- Slack ---
        slack_p = slack_in_sum - slack_out_sum
        slack_in = slack_p.copy()
        slack_in[slack_in > 0] = 0
        slack_out = slack_p.copy()
        slack_out[slack_out < 0] = 0
        plt.plot(
            time_range_to_plot,
            slack_in[time_range_to_plot],
            "k:",
            lw=1.5,
            label="Slack export",
        )
        plt.plot(
            time_range_to_plot,
            slack_out[time_range_to_plot],
            "k--",
            lw=1.5,
            label="Slack import",
        )

    # --- Plot stored energy (averaged scaling) ---
    if e_bess_profiles:
        ax = plt.gca()
        twin_ax = ax.twinx()
        color = "b"

        e_sum = sum(e_bess_profiles.values())
        twin_ax.step(
            time_range_to_plot,
            e_sum[time_range_to_plot],
            "b",
            alpha=0.8,
            lw=2,
            where="mid",
            label="Stored energy",
        )
        # color the y-axis (ticks, label, spine) to match the step color
        twin_ax.set_ylabel("Stored energy [kWh]", color=color)
        twin_ax.yaxis.label.set_color(color)
        twin_ax.tick_params(axis="y", colors=color)
        twin_ax.spines["right"].set_color(color)
        ax.set_ylabel("Power [kWh]")

    # plt.ylabel("Energy flows (kWh)")
    plt.xlabel("Time (quarter-hours)")
    plt.title("Energy System Operation Overview")
    # merge legends from main axis and twin axis
    if ax or twin_ax:
        # Be defensive: some tests mock get_legend_handles_labels and may
        # return a Mock or non-iterable. Use try/except to fall back to empty lists.
        try:
            handles1, labels1 = ax.get_legend_handles_labels()
        except Exception:
            handles1, labels1 = [], []
        try:
            handles2, labels2 = twin_ax.get_legend_handles_labels()
        except Exception:
            handles2, labels2 = [], []
        if (handles1 and labels1) or (handles2 and labels2):
            ax.legend(
                list(handles1) + list(handles2),
                list(labels1) + list(labels2),
                loc="best",
                fontsize=9,
                ncol=2,
            )
    else:
        plt.legend(loc="best", fontsize=9, ncol=2)

    x_hours = np.array(list(time_range_to_plot), dtype=int)
    # choose base date (year can be parameterized); ordinal hours count from Jan 1 00:00
    base_year = 2016
    base = Timestamp(year=base_year, month=1, day=1)
    # choose tick spacing depending on total span
    span_hours = x_hours.max() - x_hours.min() if x_hours.size else 0
    if span_hours <= 24:
        tick_step = 2
    elif span_hours <= 72:
        tick_step = 3
    elif span_hours <= 24 * 14:
        tick_step = 6
    else:
        tick_step = 24

    # pick tick positions from the numeric x-axis (matching plotted x positions)
    tick_idx = np.arange(x_hours.min(), x_hours.max() + 1, tick_step, dtype=int)

    # build labels from base date + hours
    tick_labels = [(base + Timedelta(hours=int(h))).strftime("%H:00") for h in tick_idx]

    # set ticks and labels
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    # add the date as x-axis label (use the day of the first plotted hour)
    start_date = (base + Timedelta(hours=int(x_hours.min()))).date()
    ax.set_ylabel(f"Power [kW]")
    ax.set_xlabel(f"Time (hours) — date: {start_date:%Y-%m-%d}")
    plt.tight_layout()
    plt.savefig(join(output_path, "energy_system_operation.png"))
