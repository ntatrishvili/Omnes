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
from pandas import read_csv

from utils.logging_setup import get_logger


logger = get_logger(__name__)


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from typing import Optional, List
from os.path import join

from matplotlib.path import Path
from matplotlib.patches import PathPatch

def draw_battery_icon(ax, x, y, size=0.02, color="black"):
    """Draw a stylized battery symbol centered at (x, y)."""
    w, h = size, size * 0.6
    # Define battery outline + terminal using path vertices
    verts = [
        (x - w/2, y - h/2), (x + w/2, y - h/2), (x + w/2, y + h/2),
        (x + w*0.6, y + h/2), (x + w*0.6, y + h*0.7),
        (x + w*0.4, y + h*0.7), (x + w*0.4, y + h/2),
        (x - w/2, y + h/2), (x - w/2, y - h/2),  # close shape
    ]
    codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-2) + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor="none", edgecolor=color, lw=0.8, zorder=10)
    ax.add_patch(patch)

def visualize_high_voltage_day(
    net,
    results_csv: Optional[str] = None,
    results_df: Optional[pd.DataFrame] = None,
    branches: Optional[List[List[str]]] = None,
    figsize=(16, 10),
    cmap="plasma",
    base_year: int = 2016,
    output_path: Optional[str] = None,
    scenario: Optional[str] = None,
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
        hspace=0.28,
        wspace=0.38,
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
    edge_cmap = plt.get_cmap("plasma")
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
        {"loading_percent": [edge_currents.get(i, np.nan) for i in net.line.index]},
        index=net.line.index,
    )

    # Base plot
    pplot.simple_plot(
        net,
        ax=ax_net,
        bus_size=1.0,
        ext_grid_size=2,
        line_width=1.5,
        show_plot=False,
        bus_color=node_colors,
        line_color=line_colors,
        plot_loads=True,
        plot_sgens=True,
        load_size=2.0,
        sgen_size=2.0,
    )

    # --- Overlay elements (battery) ---
    batt_buses = net.sgen.loc[
        net.sgen["name"].str.contains("batt", case=False, na=False), "bus"
    ].values

    # usage inside your plotting function:
    for bus_idx in batt_buses:
        x, y = net.bus.loc[bus_idx, ["x", "y"]]
        draw_battery_icon(ax_net, x, y, size=0.05, color="black")

    # ax_net.legend(loc="upper right", fontsize="small", frameon=True)

    ax_net.set_title(
        f"Network snapshot at peak timestep {peak_idx}\nNode=U (pu), Edge=I (kA)"
    )
    ax_net.set_xlabel("Longitude")
    ax_net.set_ylabel("Latitude")

    sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    fig.colorbar(sm, ax=ax_net, fraction=0.046, pad=0.02, label="Voltage [pu]")

    # --- Branch voltage profiles ---
    highlight_colors = ["C1", "C2", "C3", "C4"]
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
        ax_branch.plot(
            range(len(idxs)),
            vals,
            marker=markers[i % len(markers)],
            color=highlight_colors[i % len(highlight_colors)],
            label=f"Branch {i+1}",
        )
        for j, name in enumerate(branch):
            ax_branch.text(j, vals[j], name, fontsize=8, ha="center", va="bottom")

    ax_branch.set_title("Voltage profile along branches")
    ax_branch.set_xlabel("Node order")
    ax_branch.set_ylabel("Voltage [pu]")
    ax_branch.legend(fontsize="small")
    ax_branch.grid(True)

    # --- Day time-series (only branch endpoints) ---
    if branches:
        day_vm = day_df[[c for c in day_df.columns if c.endswith("_vm_pu")]].astype(
            float
        )
        for i, branch in enumerate(branches):
            last_name = branch[-1]
            last_idx = bus_name_to_index.get(f"LV1.101 Bus {last_name}")
            if last_idx is not None:
                col = f"{last_idx}_vm_pu"
                if col in day_vm.columns:
                    ax_ts.plot(
                        time_index,
                        day_vm[col],
                        label=f"{last_name} (branch {i+1})",
                        color=highlight_colors[i % len(highlight_colors)],
                        lw=1.5,
                    )
    ax_ts.set_title("Voltage time series (day of peak timestep)")
    ax_ts.set_xlabel("Time [hours]")
    ax_ts.set_ylabel("Voltage [pu]")
    ax_ts.grid(True)
    ax_ts.legend()

    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=45)
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
        branches=[[4, 8, 11, 10, 3], [4, 7, 12, 14, 6, 5], [4, 2, 9, 13], [4, 1]],
        scenario=scenario,
    )


if __name__ == "__main__":
    config = configparser.ConfigParser(
        allow_no_value=True, interpolation=configparser.ExtendedInterpolation()
    )
    config.read("..\\..\\config.ini")
    results = read_csv(
        join(
            config.get("path", "output"), "energy_system_power_flow_results_default.csv"
        )
    )
    net = simbench.get_simbench_net("1-LV-rural1--0-sw")
    visualize_high_voltage_day(
        net,
        results_df=results,
        branches=[[4, 8, 11, 10, 3], [4, 7, 12, 14, 6, 5], [4, 2, 9, 13], [4, 1]],
    )
