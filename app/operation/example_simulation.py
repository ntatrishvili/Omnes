"""
Example simulation module for energy system power flow analysis.

This module demonstrates how to run time-stepped power flow simulations on a
pandapower network. It provides utilities to set load and generator values
from time-series profiles, run power flow calculations, and collect results
for each timestep.

Sign Conventions:
    - Loads: p_mw is positive for consumption (kW → MW ÷ 1000)
    - Generators (PV/wind/battery): p_mw is negative for injection (generation)
      into the network (i.e., sgen.p_mw = -gen_kW ÷ 1000)

This sign convention is common in pandapower but should be verified for
consistency with your specific use case.
"""

import configparser
import pandapower as pp
import simbench
from matplotlib.lines import Line2D
from pandas import read_csv
from matplotlib.patches import PathPatch, Patch, Arrow
from utils.logging_setup import get_logger


logger = get_logger(__name__)


import json
import numpy as np
import matplotlib as mpl
import matplotlib.dates as mdates
from typing import Optional
from os.path import join

from matplotlib.path import Path
import pandas as pd

import matplotlib.pyplot as plt
from typing import List, Union, Tuple

def plot_branch_voltage_heatmaps(
    results: List[Union[str, pd.DataFrame]],
    branch_buses: List,
    scenario_names: List[str] = None,
    cmap: str = "Greys",
    figsize: Tuple[float, float] = (5.5, 10),
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
    mean_arr = np.full((n_scenarios, n_buses), np.nan, dtype=float)
    min_arr = np.full((n_scenarios, n_buses), np.nan, dtype=float)

    for i, res in enumerate(results):
        # load DataFrame
        if isinstance(res, str):
            df = pd.read_csv(res)
        elif isinstance(res, pd.DataFrame):
            df = res.copy()
        else:
            raise ValueError("Each item of results must be a filepath or a pandas DataFrame")

        # filter branch buses and aggregate
        branch_bus_indices = net.bus[net.bus.name.str.split(" ").str[-1].astype(int).isin(branch_buses)].index.astype(str).tolist()
        df_branch = df[[c for c in df.columns if "vm_pu" in c and c.split("_")[0] in branch_bus_indices]]
        if df_branch.empty:
            # leave NaNs if nothing found
            continue

        # build series for each bus in requested order
        for j, bus in enumerate(branch_bus_indices):
            try:
                s = df_branch[f"{bus}_vm_pu"]
            except:
                continue
            max_arr[i, j] = s.max(skipna=True)
            mean_arr[i, j] = s.mean(skipna=True)
            min_arr[i, j] = s.min(skipna=True)

    # prepare scenario names
    if scenario_names is None:
        scenario_names = [f"S{i+1}" for i in range(n_scenarios)]
    if len(scenario_names) != n_scenarios:
        raise ValueError("Length of scenario_names must match number of results")

    # plotting
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    titles = ["Node voltages - maximum values", "Node voltages - average values", "Node voltages - minimum values"]
    data_list = [max_arr, mean_arr, min_arr]
    vmin = np.nanmin(min_arr)
    vmax = np.nanmax(max_arr)
    # fallback if all NaN
    if np.isnan(vmin) or np.isnan(vmax):
        vmin, vmax = 0.0, 1.0
    for ax, data, title in zip(axes, data_list, titles):
        # im = ax.imshow(data, cmap=cmap, aspect="equal", origin="lower", vmin=vmin, vmax=vmax)
        im = ax.imshow(data, cmap=cmap, aspect="equal", origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
        im.set_zorder(1)

        # --- grid between heatmap cells ---
        # minor ticks at cell boundaries
        ax.set_xticks(np.arange(-0.5, n_buses, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_scenarios, 1), minor=True)
        # draw minor grid (cell separators)
        ax.grid(which="minor", color="lightgrey", linestyle="-", linewidth=0.6, zorder=2, alpha=0.9)
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
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
        # nice formatting for colorbar label (optional)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_ylabel("Voltage [pu]")

    # x ticks for busbars on bottom subplot only
    axes[-1].set_xticks(np.arange(n_buses))
    axes[-1].set_xticklabels(branch_buses)
    axes[-1].set_xlabel("Buses")

    plt.tight_layout()
    if savepath:
        plt.savefig(join(savepath, "ranch_voltages.png"), dpi=400, bbox_inches="tight")
    plt.show()
    return fig, axes


def fit_network_axis(ax, net, pad=0.05, min_span=1e-6):
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
    pad_x = x_span * pad
    pad_y = y_span * pad
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale(enable=False)
    ax.margins(0)


def annotate_buses(ax, net, label="name", fontsize=15, offset=(0.0, 0.0), bbox=True, filter_fn=None):
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
            txt = row.get("name", str(idx)).replace("LV1.101 Bus ", "").replace("MV1.101 Bus 4", "MV grid")
        elif label == "index":
            txt = str(idx)
        else:
            txt = str(row.get(label, ""))
        bbox_kw = dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.2) if bbox else None
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

def draw_battery_icon(ax, x, y, size=0.02, color="red"):
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
    patch = PathPatch(path, facecolor=color, edgecolor="black", lw=1.0, zorder=30, alpha=0.9)
    ax.add_patch(patch)
    # small inner terminal line for contrast
    ax.plot([x + w * 0.45, x + w * 0.6], [y + h * 0.5, y + h * 0.6], color="black", lw=1.0, zorder=31)

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
    trafo = 4
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
            return (None, None)
        try:
            geo = json.loads(geo_str)
            lon, lat = geo["coordinates"]
            return lon, lat
        except Exception:
            return (None, None)

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
        {"loading_percent": [edge_currents.get(i, np.nan)*1000 for i in net.line.index]},
        index=net.line.index,
    )

    # Base plot
    pplot.simple_plot(
        net,
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
    annotate_buses(ax_net, net, label="name", fontsize=12, offset=(0.00025, -0.0001))
    # --- Overlay elements (battery) ---
    batteries = net.sgen.loc[net.sgen["name"] == "battery"]
    battery_buses = net.bus.loc[batteries["bus"]]
    for i, row in batteries.iterrows():
        if not row.get("in_service", True):
            continue
        try:
            x, y = net.bus.loc[row["bus"], ["x", "y"]]
        except Exception:
            continue
        draw_battery_icon(ax_net, x, y, size=0.04, color="blue")

    # --- Create a small legend inside the network axes using proxy artists ---
    # PV proxy: circle + horizontal line (tuple), handled by HandlerTuple
    pv_circle = Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="white", markeredgecolor="k",
                       markersize=8, linestyle="None")
    pv_hline = Line2D([-0.2, 0.2], [0, 0], color="k", linewidth=1)

    pv_proxy = (pv_circle, pv_hline)

    # Load proxy: down-pointing triangle
    load_proxy = Line2D([0], [0], marker="v", color="k",
                        markerfacecolor="white", markeredgecolor="k",
                        markersize=8, linestyle="None")

    battery_proxy = Patch(facecolor="blue", edgecolor="k")
    extgrid_proxy = Line2D([0], [0], marker="s", color="gold", markerfacecolor="gold", markersize=8, linestyle="None")

    legend_handles = [pv_proxy, load_proxy, battery_proxy, extgrid_proxy]
    legend_labels = ["PV", "Load", "Battery", "External grid"]
    ax_net.legend(legend_handles, legend_labels, loc="lower right", framealpha=0.9, fontsize="small")

    ax_net.set_title(
        f"Node color=Voltage [pu], Edge color=Current [A]"
    )
    ax_net.set_xlabel("Longitude")
    ax_net.set_ylabel("Latitude")

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax_net)
    cax1 = divider.append_axes("right", size="3%", pad=0.05)
    cax2 = divider.append_axes("right", size="3%", pad=0.82)
    fig.colorbar(sm, cax=cax1, label="Voltage [pu]")
    cm = mpl.cm.ScalarMappable(cmap=edge_cmap, norm=mpl.colors.Normalize(vmin=cmin, vmax=cmax))
    cm.set_array([])
    fig.colorbar(cm, cax=cax2, label="Current [A]")

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
            label=f"Branch {i+1}",
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
        if bess_idx:
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
            ax_branch.text(j, vals[j]+0.0001, name, fontsize=12, ha="center", va="bottom")

    ax_branch.set_title("Voltage profile along branches")
    ax_branch.set_xlabel("Node order")
    ax_branch.set_ylabel("Voltage [pu]")
    # ax_branch.set_ylim(0.99, 1.02)
    ax_branch.legend(fontsize="small")
    ax_branch.grid(True)

    # --- Day time-series (only branch endpoints) ---
    if branches:
        day_vm = day_df[[c for c in day_df.columns if c.endswith("_vm_pu")]].astype(
            float
        )
        for i, branch in enumerate(branches+[[trafo]]):
            last_name = branch[-1]
            last_idx = bus_name_to_index.get(f"LV1.101 Bus {last_name}")
            if last_idx is not None:
                col = f"{last_idx}_vm_pu"
                if col in day_vm.columns:
                    postfix = f"(branch {i + 1} end)" if i != len(branches) else "(transformer)"
                    ax_ts.plot(
                        time_index,
                        day_vm[col],
                        label=f"Bus {last_name} {postfix}",
                        color=highlight_colors[i % len(highlight_colors)],
                        lw=1.5,
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
    fig.suptitle(f"Network snapshot at {peak_ts:%Y-%m-%d %H:00}\n{title_text}", fontsize=16)
    plt.tight_layout()

    if output_path:
        plt.savefig(
            join(output_path, f"high_voltage_day_visualization{scenario}.png"), dpi=500
        )

    plt.show()


def set_timestep_values(net: pp.pandapowerNet, timestep_idx: int):
    """
    Set load and generator power values for a specific timestep.

    Updates the pandapower network with time-series values from the profiles
    DataFrames for a given timestep. Loads are set with positive p_mw values
    (consumption), while generators (sgen) are set to inject power (convention
    may vary).

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network to update. Must have 'profiles' dict with
        'load' and 'renewables' DataFrames, indexed by timestep.
    timestep_idx : int
        The timestep index to retrieve values for. Used to index into
        net.profiles['load'] and net.profiles['renewables'].

    Notes
    -----
    - Expects net.profiles['load'] columns named as: '{load_name}_p_cons', '{load_name}_q_cons'
    - Expects net.profiles['renewables'] columns named as: '{sgen_name}_p_out'
    - Values in profiles are in kW; converted to MW for pandapower (÷ 1000)
    - Reactive power for sgens is set to 0

    Raises
    ------
    KeyError
        If timestep_idx is not in the profiles index or if expected column
        names are missing from the profiles DataFrames.
    """

    # loads
    for idx, row in net.load.iterrows():
        load_name = f"{row['name']}"
        net.load.at[idx, "p_mw"] = (
            net.profiles["load"].loc[timestep_idx, f"{load_name}_p_cons"] / 1000.0
        )
        net.load.at[idx, "q_mvar"] = (
            net.profiles["load"].loc[timestep_idx, f"{load_name}_q_cons"] / 1000.0
        )

    # sgens (pv, wind, battery)
    for idx, row in net.sgen.iterrows():
        name = f"{row['name']}"
        if name == "battery":
            net.sgen.at[idx, "p_mw"] = (
                    (net.profiles["storage"].loc[timestep_idx, f"{name}_p_out"]
                     - net.profiles["storage"].loc[timestep_idx, f"{name}_p_in"]) / 1000.0
            )
        else:
            net.sgen.at[idx, "p_mw"] = (
                net.profiles["renewables"].loc[timestep_idx, f"{name}_p_out"] / 1000.0
            )
        net.sgen.at[idx, "q_mvar"] = 0


def collect_results(net, results, time_idx):
    """
    Collect power flow results from the network and store in a results DataFrame.

    After a successful power flow calculation, this function extracts voltage
    and current results from the network and stores them in the provided
    results DataFrame.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network with computed results in net.res_bus and
        net.res_line tables.
    results : pd.DataFrame
        The results DataFrame indexed by timestep, with columns named as
        '{element_idx}_{quantity}' (e.g., '0_vm_pu', '1_i_ka'). This
        DataFrame is modified in-place.
    time_idx : int
        The row index (timestep) at which to store results in the DataFrame.

    Notes
    -----
    - Extracts bus-level quantities: vm_pu, va_degree, p_mw, q_mvar
    - Extracts line-level quantities: i_ka (current in kA)
    - Results are indexed by bus index and line index within the network

    Returns
    -------
    None
        Results are stored in-place in the provided DataFrame.
    """

    # loads
    for bus_idx, row in net.res_bus.iterrows():
        for quantity in ["vm_pu", "va_degree", "p_mw", "q_mvar"]:
            results.at[time_idx, f"{bus_idx}_{quantity}"] = row[quantity]

    # sgens (pv, wind, battery)
    for line_idx, row in net.res_line.iterrows():
        results.at[time_idx, f"{line_idx}_i_ka"] = row["i_ka"]


def init_results_frame(net, time_set):
    """
    Initialize an empty DataFrame to store power flow results for all timesteps.

    Creates a results DataFrame with one row per timestep and columns for
    each bus quantity and line quantity that will be collected after running
    power flow calculations.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network whose bus and line indices are used to generate
        column names.
    time_set : list or range
        The set of timesteps (indices) for which results will be collected.
        Used as the index of the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        An empty DataFrame with time_set as index and columns named as:
        - '{bus_idx}_vm_pu', '{bus_idx}_va_degree', '{bus_idx}_p_mw', '{bus_idx}_q_mvar'
          for each bus in net.bus
        - '{line_idx}_i_ka' for each line in net.line

    Notes
    -----
    The returned DataFrame is populated later by collect_results() after
    each power flow calculation.
    """

    columns = []
    for bus_idx, row in net.bus.iterrows():
        columns.append(f"{bus_idx}_vm_pu")
        columns.append(f"{bus_idx}_va_degree")
        columns.append(f"{bus_idx}_p_mw")
        columns.append(f"{bus_idx}_q_mvar")

        # sgens (pv, wind, battery)
    for line_idx, row in net.line.iterrows():
        columns.append(f"{line_idx}_i_ka")

    return pd.DataFrame(index=time_set, columns=columns)


def simulate_energy_system(net, **kwargs):
    """
    Run a time-stepped power flow simulation on the pandapower network.

    Iterates through all timesteps in the network's time_set, updates load
    and generator values from profiles for each timestep, runs a power flow
    calculation, and collects results. Non-convergence is logged as a warning
    but does not halt the simulation.

    Parameters
    ----------
    net : pp.pandapowerNet
        The pandapower network to simulate. Must have:
        - 'time_set' attribute containing timestep indices
        - 'profiles' dict with 'load' and 'renewables' DataFrames
        - All standard pandapower tables (bus, load, line, sgen, ext_grid, etc.)

    Returns
    -------
    None
        Results are written to a CSV file 'energy_system_power_flow_results.csv'
        in the current working directory.

    Notes
    -----
    Power flow solver settings (Newton-Raphson method, tolerance, etc.) are
    hardcoded. Modify the pp.runpp() call to change solver parameters.

    Convergence:
        - Uses 'auto' initialization for the first timestep, 'results' for
          subsequent timesteps (warm start)
        - Maximum 30 iterations allowed per timestep
        - Failure to converge is logged but the simulation continues

    Raises
    ------
    AttributeError
        If net does not have a 'time_set' attribute or if profiles are missing.
    """
    config = configparser.ConfigParser(
        allow_no_value=True, interpolation=configparser.ExtendedInterpolation()
    )
    config.read("..\\config.ini")
    time_set = net.get("time_set", [])
    results = init_results_frame(net, time_set)

    for t in time_set:
        set_timestep_values(net, t)
        try:
            pp.runpp(
                net,
                algorithm="nr",
                init="results" if t > 0 else "auto",
                tolerance_mva=1e-5,
                max_iteration=30,
                trafo_model="t",
                enforce_q_lims=False,
                calculate_voltage_angles=True,
                numba=True,
                recycle=True,
                check_connectivity=True,
            )

            if t % 1500 == 0:
                logger.info(f"Power flow at time {t} successful.")
            collect_results(net, results, t)
        except pp.LoadflowNotConverged:
            logger.warning(f"Power flow did not converge at {t}.")

    scenario = kwargs.get("scenario", "default")
    results.to_csv(
        join(
            config.get("path", "output"),
            f"energy_system_power_flow_results{scenario}.csv",
        )
    )
    visualize_high_voltage_day(
        net,
        results_df=results,
        branches=[[4, 1], [4, 2, 9, 13], [4, 7, 12, 14, 6, 5], [4, 8, 11, 10, 3]],
        scenario=scenario,
        output_path=config.get("path", "output"),
    )


if __name__ == "__main__":
    config = configparser.ConfigParser(
        allow_no_value=True, interpolation=configparser.ExtendedInterpolation()
    )
    config.read("..\\..\\config.ini")
    results = read_csv(
        join(
            config.get("path", "output"), "energy_system_power_flow_resultsdefault.csv"
        )
    )
    net = simbench.get_simbench_net("1-LV-rural1--0-sw")
    pp.create_sgen(net, bus=10, p_mw=0.0, q_mvar=0.0, name="battery")
    visualize_high_voltage_day(
        net,
        results_df=results,
        branches=[[4, 8, 11, 10, 3], [4, 7, 12, 14, 6, 5], [4, 2, 9, 13], [4, 1]],
    )

    scenarios = ["No battery", "Feeder", "Branch 1 end", "Branch 2 end", "Branch 3 end", "Branch 4 end"]
    results = []
    for sc in scenarios:
        results.append(read_csv(
            join(
                config.get("path", "output"), f"energy_system_power_flow_results{sc}.csv"
            )
        ))
    branch_buses = [4, 7, 12, 14, 6, 5]
    plot_branch_voltage_heatmaps(results, branch_buses, scenarios, cmap="RdYlBu_r",
                                 savepath=config.get("path", "output"))
