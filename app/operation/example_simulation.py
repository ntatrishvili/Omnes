# TODO: Plotting function goes here
from app.model.model import Model
import pandapower as pp


def set_timestep_values(net: pp.pandapowerNet, timestep_idx: int):
    """
    For each load/sgens set p_mw according to the model time-series values (kW -> MW).
    We use convention:
        - Loads -> create_load.p_mw is positive consumption (kW -> MW /1000)
        - Generators (pv/wind/battery) -> sgen.p_mw is NEGATIVE for injection into network
          (i.e., sgen.p_mw = -gen_kW/1000)
      (This sign convention is common but please ensure consistency in your analysis.)
    """
    # set loads
    from app.model.load.load import Load as OmnesLoad
    from app.model.generator.pv import PV as OmnesPV
    from app.model.generator.wind_turbine import Wind as OmnesWind
    from app.model.storage.battery import Battery as OmnesBattery

    # loads
    for idx, row in net.load.iterrows():
        load_name = f"load_{row['id']}"
        net.load.at[idx, "p_mw"] = (
            net.profiles["load"][f"{load_name}.p_cons"][timestep_idx] / 1000.0
        )
        net.load.at[idx, "q_mvar"] = (
            net.profiles["load"][f"{load_name}.q_cons"] / 1000.0
        )

    # sgens (pv, wind, battery)
    for idx, row in net.sgen.iterrows():
        pv_name = f"pv_{row['id']}"
        net.load.at[idx, "p_mw"] = (
            net.profiles["renewables"][f"{pv_name}.p_cons"][timestep_idx] / 1000.0
        )
        net.sgen.at[idx, "p_mw"] = -net.profiles["renewables"][f"{pv_name}.p"] / 1000.0


def simulate_energy_system(net):
    time_set = net.get("time_set", [])
    for t in time_set:
        set_timestep_values(net, t)

        # run power flow
        pp.runpp(net)
