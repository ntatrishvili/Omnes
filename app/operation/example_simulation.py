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


def visualize_high_voltage_day(
    net,
    results_csv: Optional[str] = None,
    results_df: Optional[pd.DataFrame] = None,
    branches: Optional[List[List[int]]] = None,
    figsize=(16, 10),
    cmap="viridis",
):
    """
    Visualize network at timestep with highest voltage and show branch profiles + day time-series.

    Parameters
    ----------
    net : pandapowerNet
        Pandapower network (used for topology and node positions).
    results_csv : str, optional
        Path to `energy_system_power_flow_results.csv`. Either this or results_df must be provided.
    results_df : pd.DataFrame, optional
        Results DataFrame (index = timesteps). If provided, used instead of reading CSV.
    branches : list of list of int, optional
        Predefined branches (each branch is ordered list of bus indices). If None the function tries to detect
        up to three main branches automatically.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for node voltages.
    """
    if results_df is None and results_csv is None:
        raise ValueError("Provide results_df or results_csv")

    # load results
    df = results_df if results_df is not None else pd.read_csv(results_csv, index_col=0)

    # identify vm_pu columns and line current columns
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

    # compute per-timestep maximum bus voltage
    vm_df = df[vm_cols].astype(float)
    max_vm_per_t = vm_df.max(axis=1)
    # find the timestep with the highest maximum voltage
    peak_idx = max_vm_per_t.idxmax()

    # attempt to interpret index as datetime to get the full day
    try:
        parsed_index = pd.to_datetime(df.index, errors="coerce")
        if parsed_index.notna().sum() > 0:
            df_dt = df.copy()
            df_dt.index = parsed_index
            peak_ts = parsed_index[df.index.get_indexer([peak_idx])[0]]
            peak_date = peak_ts.date()
            # select all rows with same date
            day_mask = df_dt.index.date == peak_date
            day_df = df_dt.loc[day_mask]
        else:
            raise ValueError
    except Exception:
        # fallback: assume hourly integer timesteps, group by day = t // 24
        numeric_idx = pd.to_numeric(df.index, errors="coerce")
        if numeric_idx.isna().all():
            # cannot parse index -> use single timestep only
            peak_ts = peak_idx
            day_df = df.loc[[peak_idx]]
        else:
            day_number = int(numeric_idx[df.index.get_indexer([peak_idx])[0]] // 24)
            start = day_number * 24
            end = start + 24
            day_mask = (numeric_idx >= start) & (numeric_idx < end)
            day_df = df.loc[day_mask]

    # prepare positions for network layout (prefer stored coords)
    pos = {}
    if "x" in net.bus.columns and "y" in net.bus.columns:
        for idx, r in net.bus.iterrows():
            pos[idx] = (r["x"], r["y"])
    elif (
        hasattr(net, "bus_geodata")
        and "x" in net.bus_geodata.columns
        and "y" in net.bus_geodata.columns
    ):
        for idx, r in net.bus_geodata.iterrows():
            pos[idx] = (r["x"], r["y"])
    else:
        G_tmp = nx.Graph()
        G_tmp.add_nodes_from(net.bus.index.tolist())
        pos = nx.circular_layout(G_tmp)

    # build topology graph from net.line (and line with from_bus/to_bus)
    G = nx.Graph()
    G.add_nodes_from(net.bus.index.tolist())
    for idx, row in net.line.iterrows():
        u = int(row["from_bus"])
        v = int(row["to_bus"])
        G.add_edge(u, v, key=idx)

    # determine branches if not provided: find root (ext_grid.bus if present) then longest root->leaf paths
    if branches is None:
        # choose root as ext_grid bus if available, else pick bus with highest degree
        root = None
        if (
            hasattr(net, "ext_grid")
            and not net.ext_grid.empty
            and "bus" in net.ext_grid.columns
        ):
            root = int(net.ext_grid.iloc[0]["bus"])
        if root is None:
            degrees = dict(G.degree())
            # prefer node with degree > 1 and smallest index
            root = min(degrees, key=lambda k: (-degrees[k], k))

        leaves = [n for n, d in G.degree() if d == 1 and n != root]
        # compute root->leaf paths and sort by length
        paths = []
        for leaf in leaves:
            try:
                path = nx.shortest_path(G, source=root, target=leaf)
                paths.append(path)
            except nx.NetworkXNoPath:
                continue
        paths.sort(key=lambda p: len(p), reverse=True)
        # take up to 3 longest distinct paths
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

    # collect vm_pu at peak timestep for node coloring
    peak_vm = {}
    # peak_idx may be datetime or original label; use df.loc
    peak_row = df.loc[peak_idx]
    for c in vm_cols:
        bus = int(c.split("_")[0])
        peak_vm[bus] = float(peak_row[c]) if pd.notna(peak_row[c]) else np.nan

    # collect edge currents at peak timestep
    edge_currents = {}
    for idx, row in net.line.iterrows():
        u = int(row["from_bus"])
        v = int(row["to_bus"])
        col_name = f"{idx}_i_ka"
        i_val = (
            float(peak_row[col_name])
            if col_name in df.columns and pd.notna(peak_row[col_name])
            else np.nan
        )
        edge_currents[(u, v, idx)] = i_val

    # plotting layout: 2x2 (top row two panels, bottom full width)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.3, 1.0], hspace=0.28, wspace=0.3
    )
    ax_net = fig.add_subplot(gs[0, 0])
    ax_branch = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[1, :])

    # Draw edges with widths proportional to current
    currents = np.array(
        [v for v in edge_currents.values() if not np.isnan(v)], dtype=float
    )
    if currents.size:
        cur_min, cur_max = currents.min(), currents.max()
    else:
        cur_min, cur_max = 0.0, 1.0

    widths = []
    edge_colors = []
    for u, v, idx in edge_currents:
        val = edge_currents[(u, v, idx)]
        if np.isnan(val):
            widths.append(0.6)
            edge_colors.append("lightgray")
        else:
            w = (
                0.6 + 5.4 * ((val - cur_min) / (cur_max - cur_min))
                if cur_max > cur_min
                else 1.5
            )
            widths.append(w)
            cmap_obj = plt.cm.plasma
            normed = (val - cur_min) / (cur_max - cur_min) if cur_max > cur_min else 0.5
            edge_colors.append(cmap_obj(normed))

    nx.draw_networkx_edges(
        G, pos=pos, ax=ax_net, edge_color=edge_colors, width=widths, alpha=0.85
    )

    # node colors by peak vm
    node_list = list(G.nodes())
    node_vals = [peak_vm.get(n, np.nan) for n in node_list]
    vmin = np.nanmin(node_vals) if np.isfinite(node_vals).any() else 0.9
    vmax = np.nanmax(node_vals) if np.isfinite(node_vals).any() else 1.1
    nodes_drawn = nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=node_list,
        node_size=300,
        node_color=node_vals,
        cmap=plt.get_cmap(cmap),
        vmin=vmin,
        vmax=vmax,
        ax=ax_net,
    )

    labels = {}
    if "name" in net.bus.columns:
        for n in node_list:
            labels[n] = str(net.bus.at[n, "name"])
    else:
        for n in node_list:
            labels[n] = str(n)
    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=8, ax=ax_net)

    sm = mpl.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap), norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_net, fraction=0.046, pad=0.02)
    cbar.set_label("Voltage [pu]")

    ax_net.set_title(
        f"Network snapshot at peak timestep {peak_idx}\n(node color = vm_pu, edge width = i_ka)"
    )

    # Branch spatial profiles (vm along node-order)
    for i, branch in enumerate(branches):
        branch_v = [peak_vm.get(b, np.nan) for b in branch]
        x = list(range(len(branch)))
        ax_branch.plot(x, branch_v, marker="o", label=f"Branch {i+1}: root→leaf")
        for xi, b in zip(x, branch):
            ax_branch.text(
                xi, branch_v[xi], str(b), fontsize=8, ha="center", va="bottom"
            )
    ax_branch.set_xlabel("Node order along branch (root → leaf)")
    ax_branch.set_ylabel("Voltage [pu]")
    ax_branch.set_title("Voltage profile along main branches (at peak timestep)")
    ax_branch.grid(True)
    ax_branch.legend(fontsize="small")

    # Time series for the day containing the peak: vm for each bus
    # Determine vm columns for day_df
    day_vm = day_df[[c for c in day_df.columns if c.endswith("_vm_pu")]].astype(float)
    # use parsed datetime index if available
    try:
        time_index = pd.to_datetime(day_df.index)
    except Exception:
        time_index = day_df.index

    # plot each bus time series (keep light colors for many lines)
    for c in day_vm.columns:
        bus = int(c.split("_")[0])
        ax_ts.plot(time_index, day_vm[c].values, label=str(bus), alpha=0.7, linewidth=1)
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("Voltage [pu]")
    ax_ts.set_title(f"Voltage time-series for day of peak (peak timestep: {peak_idx})")
    ax_ts.grid(True)
    # place legend only if not too many lines
    if len(day_vm.columns) <= 12:
        ax_ts.legend(title="bus", fontsize="small", ncol=2)

    plt.tight_layout()
    plt.show()


logger = get_logger(__name__)


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
        net.load.at[idx, "p_mw"] = net.profiles["load"].loc[
            timestep_idx, f"{load_name}_p_cons"
        ]
        net.load.at[idx, "q_mvar"] = net.profiles["load"].loc[
            timestep_idx, f"{load_name}_q_cons"
        ]

    # sgens (pv, wind, battery)
    for idx, row in net.sgen.iterrows():
        name = f"{row['name']}"
        if name == "battery":
            continue
        net.sgen.at[idx, "p_mw"] = net.profiles["renewables"].loc[
            timestep_idx, f"{name}_p_out"
        ]
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
