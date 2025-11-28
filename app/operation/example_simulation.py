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

from copy import copy

import pandapower as pp
import simbench
from pandas import read_csv

from app.infra.configuration import Config
from app.infra.logging_setup import get_logger
from app.infra.visualize import (
    plot_losses_violations_heatmaps,
    visualize_high_voltage_day,
)

logger = get_logger(__name__)

from os.path import join

import pandas as pd


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
                net.profiles["storage"].loc[timestep_idx, f"{name}_p_out"]
                - net.profiles["storage"].loc[timestep_idx, f"{name}_p_in"]
            ) / 1000.0
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
            Config().get("path", "output"),
            f"energy_system_power_flow_results{scenario}.csv",
        )
    )
    visualize_high_voltage_day(
        net,
        results_df=results,
        branches=[[4, 1], [4, 2, 9, 13], [4, 7, 12, 14, 6, 5], [4, 8, 11, 10, 3]],
        scenario=scenario,
        output_path=Config().get("path", "output"),
    )


if __name__ == "__main__":
    net = simbench.get_simbench_net("1-LV-rural1--0-sw")
    scenarios = [
        "No battery",
        "Feeder",
        "Branch 1 end",
        "Branch 2 end",
        "Branch 3 end",
        "Branch 4 end",
    ]
    buses = [None, 4, 1, 13, 5, 3]
    results = []
    for sc, bess_bus in zip(scenarios, buses):
        res_df = read_csv(
            join(
                Config().get("path", "output"),
                f"energy_system_power_flow_results{sc}.csv",
            )
        )

        pnet = copy(net)
        if bess_bus is not None:
            pp.create_sgen(
                pnet,
                bus=net.bus.loc[net.bus.name.str.contains(f"{bess_bus}")].index[0],
                p_mw=0.0,
                q_mvar=0.0,
                name="battery",
            )
        # visualize_high_voltage_day(
        #     pnet,
        #     results_df=res_df,
        #     branches=[[4, 1], [4, 2, 9, 13], [4, 7, 12, 14, 6, 5], [4, 8, 11, 10, 3]],
        #     scenario=sc,
        #     output_path=Config().get("path", "output"),
        #     remove_battery=True
        # )
        results.append(res_df)
    branch_buses = [4, 1, 13, 5, 3]
    # bus_names = ["feeder", "branch 1", "branch 2", "branch 3", "branch 4"]
    # plot_branch_voltage_heatmaps(
    #     results,
    #     branch_buses,
    #     bus_names,
    #     scenarios,
    #     cmap="RdYlBu_r",
    #     savepath=Config().get("path", "output"),
    # )
    plot_losses_violations_heatmaps(
        results,
        net,
        scenarios,
        cmap="RdYlBu_r",
        savepath=Config().get("path", "output"),
    )
