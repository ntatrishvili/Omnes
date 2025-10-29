import pandapower as pp
import pandas as pd

from utils.logging_setup import get_logger


logger = get_logger(__name__)
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
        net.load.at[idx, "p_mw"] = net.profiles["load"].loc[timestep_idx, f"{load_name}_p_cons"]
        net.load.at[idx, "q_mvar"] = net.profiles["load"].loc[timestep_idx, f"{load_name}_q_cons"]

    # sgens (pv, wind, battery)
    for idx, row in net.sgen.iterrows():
        pv_name = f"{row['name']}"
        net.sgen.at[idx, "p_mw"] = net.profiles["renewables"].loc[timestep_idx, f"{pv_name}_p_out"]
        net.sgen.at[idx, "q_mvar"] = 0

def collect_results(net, results, time_idx):
    # loads
    for bus_idx, row in net.res_bus.iterrows():
        for quantity in ["vm_pu", "va_degree", "p_mw", "q_mvar"]:
            results.at[time_idx, f"{bus_idx}_{quantity}"] = row[quantity]

    # sgens (pv, wind, battery)
    for line_idx, row in net.res_line.iterrows():
        results.at[time_idx, f"{line_idx}_i_ka"] = row["i_ka"]


def init_results_frame(net, time_set):
    columns = []
    for bus_idx, row in net.bus.iterrows():
        columns.append(f"{bus_idx}_vm_pu")
        columns.append(f"{bus_idx}_va_degree")
        columns.append(f"{bus_idx}_p_mw")
        columns.append(f"{bus_idx}_q_mvar")

        # sgens (pv, wind, battery)
    for line_idx, row in net.line.iterrows():
        columns.append(f"{line_idx}_i_ka")

    return pd.DataFrame(
        index=time_set, columns=columns
    )


def simulate_energy_system(net):
    time_set = net.get("time_set", [])
    results = init_results_frame(net, time_set)

    # inspect trafo
    print(net.trafo.loc[0, ["vn_hv_kv", "vn_lv_kv", "sn_mva", "vk_percent", "vkr_percent"]])

    # inspect HV / LV bus nominal voltages that trafo connects to
    hv_bus = int(net.trafo.loc[0, "hv_bus"])
    lv_bus = int(net.trafo.loc[0, "lv_bus"])
    print("hv bus vn_kv:", net.bus.at[hv_bus, "vn_kv"])
    print("lv bus vn_kv:", net.bus.at[lv_bus, "vn_kv"])

    # inspect typical line impedances (compare magnitudes)
    print(net.line[["name", "r_ohm_per_km", "x_ohm_per_km", "length_km"]])

    for t in time_set:
        set_timestep_values(net, t)
        try:
            pp.runpp(net, algorithm='bfsw', init='flat')
            if t%100 == 0:
                logger.info(f"Step: {t}.")
            collect_results(net, results, t)
        except pp.LoadflowNotConverged:
            logger.info(f"Power flow did not converge at {t}.")

    results.to_csv("energy_system_power_flow_results.csv")
