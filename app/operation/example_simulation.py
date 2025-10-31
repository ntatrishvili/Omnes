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
from os.path import join

import pandapower as pp

from utils.logging_setup import get_logger

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl
from typing import List, Optional

logger = get_logger(__name__)


def visualize_high_voltage_day(
    net,
    results_csv: Optional[str] = None,
    results_df: Optional[pd.DataFrame] = None,
    branches: Optional[List[List[int]]] = None,
    figsize=(16, 10),
    cmap="viridis",
    base_year: int = 2020,
    output_path: Optional[str] = None,
):
    """
    Visualize network at timestep with highest voltage and show branch profiles + day time-series.

    Assumptions:
    - All bus geocoordinates are present in `net.bus` as columns `x` and `y`.
    - results_df or results_csv is provided (results DataFrame with columns like '0_vm_pu', '1_i_ka').
    - time indices are either datetimes or integer hour ordinals (hours since Jan 1 of some year).
    """
    from pandapower.plotting import create_bus_collection, create_line_collection
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors

    if results_df is None and results_csv is None:
        raise ValueError("Provide results_df or results_csv")

    # load results
    df = results_df if results_df is not None else pd.read_csv(results_csv, index_col=0)

    # vm and i columns
    vm_cols = sorted(
        [c for c in df.columns if c.endswith("_vm_pu")],
        key=lambda s: int(s.split("_")[0]),
    )
    i_cols = sorted(
        [c for c in df.columns if c.endswith("_i_ka")],
        key=lambda s: int(s.split("_")[0]),
    )

    if not vm_cols:
        raise RuntimeError("No vm_pu columns found in results")

    vm_df = df[vm_cols].astype(float)
    max_vm_per_t = vm_df.max(axis=1)
    peak_idx = max_vm_per_t.idxmax()
    peak_row = df.loc[peak_idx]

    # peak vm per bus (ensure bus keys match net.bus index type)
    peak_vm = {}
    for c in vm_cols:
        try:
            bus = int(c.split("_")[0])
        except Exception:
            bus = c.split("_")[0]
        peak_vm[bus] = float(peak_row[c]) if pd.notna(peak_row[c]) else np.nan

    # collect edge currents keyed by line index
    edge_currents = {}
    for line_idx, row in net.line.iterrows():
        col_name = f"{line_idx}_i_ka"
        i_val = (
            float(peak_row[col_name])
            if col_name in df.columns and pd.notna(peak_row[col_name])
            else np.nan
        )
        edge_currents[line_idx] = i_val

    # determine day_df and time_index as datetimes
    # try parse index to datetime
    parsed_index = pd.to_datetime(df.index, errors="coerce")
    if parsed_index.notna().sum() > 0:
        # use datetime-indexed day
        df_dt = df.copy()
        df_dt.index = parsed_index
        peak_pos = df.index.get_indexer([peak_idx])[0]
        peak_ts = parsed_index[peak_pos]
        peak_date = peak_ts.date()
        day_mask = df_dt.index.date == peak_date
        day_df = df_dt.loc[day_mask]
        time_index = pd.to_datetime(day_df.index)
    else:
        # fallback numeric hourly indices -> map to datetimes using base_year Jan 1
        numeric_idx = pd.to_numeric(df.index, errors="coerce")
        if numeric_idx.isna().all():
            # single timestep only
            day_df = df.loc[[peak_idx]]
            time_index = pd.to_datetime([pd.Timestamp(year=base_year, month=1, day=1)])
        else:
            peak_pos = df.index.get_indexer([peak_idx])[0]
            peak_hour = int(numeric_idx[peak_pos])
            day_number = peak_hour // 24
            start = day_number * 24
            end = start + 24
            mask = (numeric_idx >= start) & (numeric_idx < end)
            day_df = df.loc[mask]
            hours = [int(h) for h in numeric_idx[mask]]
            base = pd.Timestamp(year=base_year, month=1, day=1)
            time_index = [base + pd.Timedelta(hours=h) for h in hours]
            time_index = pd.to_datetime(time_index)

    # build topology and find branches if not provided
    G = nx.Graph()
    G.add_nodes_from(net.bus.index.tolist())
    for idx, row in net.line.iterrows():
        u = int(row["from_bus"])
        v = int(row["to_bus"])
        G.add_edge(u, v, key=idx)

    if branches is None:
        root = None
        if (
            hasattr(net, "ext_grid")
            and not net.ext_grid.empty
            and "bus" in net.ext_grid.columns
        ):
            root = int(net.ext_grid.iloc[0]["bus"])
        if root is None:
            degrees = dict(G.degree())
            root = min(degrees, key=lambda k: (-degrees[k], k))
        leaves = [n for n, d in G.degree() if d == 1 and n != root]
        paths = []
        for leaf in leaves:
            try:
                path = nx.shortest_path(G, source=root, target=leaf)
                paths.append(path)
            except nx.NetworkXNoPath:
                continue
        paths.sort(key=lambda p: len(p), reverse=True)
        branches = []
        added_sets = []
        for p in paths:
            s = frozenset(p)
            if any(s <= other for other in added_sets):
                continue
            branches.append(p)
            added_sets.append(s)
            if len(branches) >= 3:
                break

    # --- plotting ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.3, 1.0], hspace=0.28, wspace=0.3
    )
    ax_net = fig.add_subplot(gs[0, 0])
    ax_branch = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[1, :])

    # prepare node colors
    node_list = list(net.bus.index)
    node_vals = np.array(
        [
            peak_vm.get(int(n) if not isinstance(n, str) else int(n), np.nan)
            for n in node_list
        ],
        dtype=float,
    )
    cmap_obj = plt.get_cmap(cmap)
    vmin = np.nanmin(node_vals) if np.isfinite(node_vals).any() else 0.9
    vmax = np.nanmax(node_vals) if np.isfinite(node_vals).any() else 1.1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    node_colors = [
        cmap_obj(norm(v)) if pd.notna(v) else (0.8, 0.8, 0.8, 1.0) for v in node_vals
    ]

    # create bus and line collections (use provided geocoordinates)
    bus_coll = create_bus_collection(
        net, buses=node_list, size=200, color=node_colors, zorder=3
    )
    ax_net.add_collection(bus_coll)

    # build line color/width lists matched to net.line.index order
    line_indices = list(net.line.index)
    currents = np.array(
        [edge_currents.get(i, np.nan) for i in line_indices], dtype=float
    )
    finite_curr = currents[np.isfinite(currents)]
    if finite_curr.size:
        cur_min, cur_max = finite_curr.min(), finite_curr.max()
    else:
        cur_min, cur_max = 0.0, 1.0
    edge_cmap = plt.get_cmap("plasma")
    line_colors = []
    line_widths = []
    for val in currents:
        if np.isnan(val):
            line_colors.append("lightgray")
            line_widths.append(0.6)
        else:
            normed = (val - cur_min) / (cur_max - cur_min) if cur_max > cur_min else 0.5
            line_colors.append(edge_cmap(normed))
            line_widths.append(0.6 + 5.4 * normed if cur_max > cur_min else 1.5)

    line_coll = create_line_collection(
        net, lines=line_indices, color=line_colors, linewidths=line_widths, zorder=2
    )
    ax_net.add_collection(line_coll)

    # highlight branches: thicker colored line & node markers
    highlight_colors = ["C1", "C2", "C3"]
    for i, branch in enumerate(branches):
        branch_line_idxs = []
        for u, v in zip(branch[:-1], branch[1:]):
            matches = net.line[
                ((net.line["from_bus"] == u) & (net.line["to_bus"] == v))
                | ((net.line["from_bus"] == v) & (net.line["to_bus"] == u))
            ]
            if not matches.empty:
                branch_line_idxs.append(matches.index[0])
        if branch_line_idxs:
            coll = create_line_collection(
                net,
                lines=branch_line_idxs,
                color=highlight_colors[i % len(highlight_colors)],
                linewidths=4.0,
                zorder=4,
            )
            ax_net.add_collection(coll)
            coll_b = create_bus_collection(
                net,
                buses=branch,
                size=240,
                color=highlight_colors[i % len(highlight_colors)],
                zorder=5,
            )
            ax_net.add_collection(coll_b)

    # annotate bus names/indices at their geocoordinates
    import json

    def extract_coords(geo_str):
        if pd.isna(geo_str):
            return (None, None)
        try:
            geo = json.loads(geo_str)
            lon, lat = geo["coordinates"]
            return lon, lat
        except Exception:
            return (None, None)

    net.bus[["x", "y"]] = net.bus["geo"].apply(lambda g: pd.Series(extract_coords(g)))
    for b, r in net.bus.iterrows():
        x = float(r["x"])
        y = float(r["y"])
        label = (
            str(r["name"])
            if "name" in net.bus.columns and pd.notna(r.get("name"))
            else str(b)
        )
        ax_net.text(x, y, label, fontsize=7, ha="center", va="center", zorder=6)

    # set limits and aspect from geocoordinates
    xs = net.bus["x"].astype(float).values
    ys = net.bus["y"].astype(float).values
    x_margin = (xs.max() - xs.min()) * 0.06 if xs.max() != xs.min() else 1.0
    y_margin = (ys.max() - ys.min()) * 0.06 if ys.max() != ys.min() else 1.0
    ax_net.set_xlim(xs.min() - x_margin, xs.max() + x_margin)
    ax_net.set_ylim(ys.min() - y_margin, ys.max() + y_margin)
    ax_net.set_aspect("equal")

    # colorbar for node voltages
    sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_net, fraction=0.046, pad=0.02)
    cbar.set_label("Voltage [pu]")
    ax_net.set_title(
        f"Network snapshot at peak timestep {peak_idx}\n(node color = vm_pu, edge color/width = i_ka)"
    )

    # Branch spatial profiles: ensure values exist and set y-limits to make near-1.0 visible
    overall_vmin = (
        float(np.nanmin([v for v in peak_vm.values() if pd.notna(v)]))
        if any(pd.notna(list(peak_vm.values())))
        else 0.99
    )
    overall_vmax = (
        float(np.nanmax([v for v in peak_vm.values() if pd.notna(v)]))
        if any(pd.notna(list(peak_vm.values())))
        else 1.01
    )
    y_pad = max(1e-4, (overall_vmax - overall_vmin) * 0.15)

    plotted_any_branch = False
    for i, branch in enumerate(branches):
        branch_v = []
        for b in branch:
            # ensure keys are ints when necessary
            key = int(b) if not isinstance(b, str) else int(b)
            branch_v.append(peak_vm.get(key, np.nan))
        branch_v = np.array(branch_v, dtype=float)
        if np.all(np.isnan(branch_v)):
            continue
        x = np.arange(len(branch_v))
        ax_branch.plot(
            x,
            branch_v,
            marker="o",
            label=f"Branch {i+1}: root→leaf",
            color=highlight_colors[i % len(highlight_colors)],
            linewidth=1.5,
        )
        for xi, val in zip(x, branch_v):
            if pd.notna(val):
                ax_branch.text(
                    xi, val, str(branch[int(xi)]), fontsize=8, ha="center", va="bottom"
                )
        plotted_any_branch = True

    if plotted_any_branch:
        ax_branch.set_ylim((overall_vmin - y_pad, overall_vmax + y_pad))
    ax_branch.set_xlabel("Node order along branch (root → leaf)")
    ax_branch.set_ylabel("Voltage [pu]")
    ax_branch.set_title("Voltage profile along main branches (at peak timestep)")
    ax_branch.grid(True)
    if plotted_any_branch:
        ax_branch.legend(fontsize="small")

    # Time series for the day: use datetime x-axis and format ticks as '%H:00'
    day_vm = day_df[[c for c in day_df.columns if c.endswith("_vm_pu")]].astype(float)
    # ensure time_index is a pandas.DatetimeIndex
    if not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.to_datetime(time_index)
    for c in day_vm.columns:
        bus = int(c.split("_")[0])
        ax_ts.plot(time_index, day_vm[c].values, label=str(bus), alpha=0.7, linewidth=1)
    ax_ts.set_xlabel(
        f"Time (date: {(time_index[0].date() if len(time_index) else ''):%Y-%m-%d})"
    )
    ax_ts.set_ylabel("Voltage [pu]")
    ax_ts.set_title(f"Voltage time-series for day of peak (peak timestep: {peak_idx})")
    ax_ts.grid(True)

    # format x-axis ticks as hours 'HH:00'
    span_hours = (
        (time_index.max() - time_index.min()).total_seconds() / 3600.0
        if len(time_index) > 1
        else 24
    )
    if span_hours <= 24:
        major_interval = 1
    elif span_hours <= 72:
        major_interval = 3
    elif span_hours <= 24 * 14:
        major_interval = 6
    else:
        major_interval = 24

    ax_ts.xaxis.set_major_locator(mdates.HourLocator(interval=major_interval))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%H:00"))
    fig.autofmt_xdate(rotation=45)

    if len(day_vm.columns) <= 12:
        ax_ts.legend(title="bus", fontsize="small", ncol=2)

    plt.tight_layout()
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
            continue
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


def simulate_energy_system(net):
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

            if t % 100 == 0:
                logger.info(f"Power flow at time {t} successful.")
            collect_results(net, results, t)
        except pp.LoadflowNotConverged:
            logger.warning(f"Power flow did not converge at {t}.")

    results.to_csv(
        join(config.get("path", "output"), "energy_system_power_flow_results.csv")
    )
    visualize_high_voltage_day(net, results_df=results)
