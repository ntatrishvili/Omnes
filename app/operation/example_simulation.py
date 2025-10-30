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
import pandas as pd

from utils.logging_setup import get_logger

# python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl


def visualize_power_flow_results(
    net,
    results_csv=None,
    results_df=None,
    timestep=None,
    buses=None,
    lines=None,
    figsize=(14, 8),
    cmap="viridis",
):
    """
    Visualize a network snapshot and time series from a results CSV.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network (used to build topology and node positions).
    results_csv : str
        Path to the CSV produced by simulate_energy_system (index = timesteps).
    timestep : int or label, optional
        Which timestep row to visualize. If None uses last row in CSV.
    buses : list of bus indices, optional
        If provided, time series subplot will show these bus voltages.
    lines : list of line indices, optional
        If provided, only these lines are drawn; otherwise all net.line rows are used.
    figsize : tuple
        Figure size.
    cmap : str
        Matplotlib colormap for voltages.
    """
    # load results
    if results_df is None and results_csv is None:
        raise ValueError("results_df or results_csv must be provided.")

    if results_df is None:
        df = pd.read_csv(results_csv, index_col=0)
    else:
        df = results_df

    if timestep is None:
        timestep = df.index[-1]

    # Prepare node positions
    pos = {}
    if "x" in net.bus.columns and "y" in net.bus.columns:
        for idx, row in net.bus.iterrows():
            pos[idx] = (row["x"], row["y"])
    elif (
        hasattr(net, "bus_geodata")
        and "x" in net.bus_geodata.columns
        and "y" in net.bus_geodata.columns
    ):
        for idx, row in net.bus_geodata.iterrows():
            pos[idx] = (row["x"], row["y"])
    else:
        # fallback layout
        G_tmp = nx.Graph()
        G_tmp.add_nodes_from(net.bus.index.tolist())
        pos = nx.circular_layout(G_tmp)

    # Build graph and collect edge currents
    G = nx.Graph()
    for b_idx in net.bus.index:
        G.add_node(b_idx)

    edge_currents = {}
    for idx, row in net.line.iterrows():
        if lines is not None and idx not in lines:
            continue
        from_b = int(row["from_bus"])
        to_b = int(row["to_bus"])
        G.add_edge(from_b, to_b, key=idx)
        col_name = f"{idx}_i_ka"
        i_val = df.at[timestep, col_name] if col_name in df.columns else np.nan
        edge_currents[(from_b, to_b, idx)] = i_val

    # Node voltages for the snapshot
    node_v = []
    node_list = list(G.nodes())
    for n in node_list:
        col = f"{n}_vm_pu"
        node_v.append(df.at[timestep, col] if col in df.columns else np.nan)
    node_v = np.array(node_v, dtype=float)

    # Plotting
    fig, (ax_net, ax_ts) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]}
    )

    # Draw edges with widths proportional to current
    all_currents = np.array(
        [v for v in edge_currents.values() if not np.isnan(v)], dtype=float
    )
    if all_currents.size:
        cur_min, cur_max = all_currents.min(), all_currents.max()
    else:
        cur_min, cur_max = 0.0, 1.0

    # map each edge to a width and color
    widths = []
    edge_colors = []
    for u, v, idx in edge_currents:
        val = edge_currents[(u, v, idx)]
        if np.isnan(val):
            widths.append(0.5)
            edge_colors.append("gray")
        else:
            # scale widths between 0.5 and 6
            if cur_max > cur_min:
                w = 0.5 + 5.5 * ((val - cur_min) / (cur_max - cur_min))
            else:
                w = 1.5
            widths.append(w)
            edge_colors.append(
                plt.cm.plasma(
                    (val - cur_min) / (cur_max - cur_min) if cur_max > cur_min else 0.5
                )
            )

    nx.draw_networkx_edges(
        G, pos=pos, ax=ax_net, edge_color=edge_colors, width=widths, alpha=0.8
    )

    # Draw nodes colored by vm_pu
    vmin = np.nanmin(node_v) if np.isfinite(node_v).any() else 0.9
    vmax = np.nanmax(node_v) if np.isfinite(node_v).any() else 1.1
    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_size=300,
        node_color=node_v,
        cmap=plt.get_cmap(cmap),
        vmin=vmin,
        vmax=vmax,
        ax=ax_net,
    )
    # labels: prefer net.bus.name if available
    labels = {}
    if "name" in net.bus.columns:
        for n in node_list:
            labels[n] = str(net.bus.at[n, "name"])
    else:
        for n in node_list:
            labels[n] = str(n)
    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=8, ax=ax_net)

    # Colorbar for node voltages
    sm = mpl.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap), norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_net, fraction=0.046, pad=0.04)
    cbar.set_label("Voltage [pu]")

    # Edge current legend (min/max)
    ax_net.set_title(
        f"Network snapshot at timestep {timestep}\n(node color = vm_pu, edge width = i_ka)"
    )

    # Time series subplot: voltages for selected buses (default: first 6 buses)
    if buses is None:
        buses = list(net.bus.index[:6])
    ts_df = df.copy()
    ts_v = {}
    for b in buses:
        col = f"{b}_vm_pu"
        if col in ts_df.columns:
            ts_v[b] = ts_df[col].astype(float)
    if ts_v:
        for b, series in ts_v.items():
            ax_ts.plot(series.index, series.values, label=str(b))
        ax_ts.set_xlabel("Timestep")
        ax_ts.set_ylabel("Voltage [pu]")
        ax_ts.set_title("Bus voltages over time")
        ax_ts.legend(fontsize="small")
        ax_ts.grid(True)
    else:
        ax_ts.text(0.5, 0.5, "No bus vm_pu time series found", ha="center", va="center")
        ax_ts.set_axis_off()

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
        pv_name = f"{row['name']}"
        net.sgen.at[idx, "p_mw"] = net.profiles["renewables"].loc[
            timestep_idx, f"{pv_name}_p_out"
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
    visualize_power_flow_results(net, results_df=results, buses=net.bus.index.tolist())
