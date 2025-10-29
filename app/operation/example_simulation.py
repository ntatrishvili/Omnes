# TODO: Plotting function goes here
import pandapower as pp
import pandas as pd


def set_timestep_values(net: pp.pandapowerNet, timestep_idx: int):
    """
    For each load/sgens set p_mw according to the model time-series values (kW -> MW).
    We use convention:
        - Loads -> create_load.p_mw is positive consumption (kW -> MW /1000)
        - Generators (pv/wind/battery) -> sgen.p_mw is NEGATIVE for injection into network
          (i.e., sgen.p_mw = -gen_kW/1000)
      (This sign convention is common but please ensure consistency in your analysis.)
    """

    # loads
    for idx, row in net.load.iterrows():
        load_name = f"{row['name']}"
        net.load.at[idx, "p_mw"] = net.profiles["load"][f"{load_name}_p_cons"][
            timestep_idx
        ]
        net.load.at[idx, "q_mvar"] = net.profiles["load"][f"{load_name}_q_cons"][
            timestep_idx
        ]

    # sgens (pv, wind, battery)
    for idx, row in net.sgen.iterrows():
        pv_name = f"{row['name']}"
        net.sgen.at[idx, "p_mw"] = net.profiles["renewables"][f"{pv_name}_p_out"][
            timestep_idx
        ]

    net.load.to_csv("load_debug.csv")
    net.sgen.to_csv("sgen_debug.csv")
    net.bus.to_csv("bus_debug.csv")
    net.line.to_csv("line_debug.csv")


def collect_results(net, results, time_idx):
    # loads
    for bus_idx, row in net.res_bus.iterrows():
        results.at[time_idx, bus_idx] = row["vm_pu"]

    # sgens (pv, wind, battery)
    for line_idx, row in net.line.iterrows():
        results.at[time_idx, line_idx] = row["i_ka"]


def simulate_energy_system(net):
    time_set = net.get("time_set", [])
    results = pd.DataFrame(
        index=time_set, columns=net.bus.index.tolist() + net.line.index.tolist()
    )
    for t in time_set:
        set_timestep_values(net, t)
        try:
            # run power flow
            pp.runpp(net)
            collect_results(net, results, t)
        except:
            pass

    results.to_csv("energy_system_power_flow_results.csv")
