
import pulp

# from ..model.consumer import Consumer
# from ..model.pv import PV
# from ..model.battery import Battery
# from ..model.slack import Slack


def create_empty_pulp_var(name: str, time_set: int) -> list[pulp.LpVariable]:
    """
    Create a list of empty LpVariable with the specified name and time set
    """
    return [pulp.LpVariable(f"P_{name}_{t}", lowBound=0) for t in range(time_set)]


# def convert() -> tuple[list, dict]:
#     production = PV().production

#     max_power_bess = production.max()
#     max_stored_energy_bess = max_power_bess * 8

#     # Time interval of the optimization
#     time_set = len(production)

#     p_cons = Consumer().consumption.values
#     p_pv = production.values

#     slack = Slack()
#     # Energy injected into the household by the slack
#     p_slack_out = slack.get_flow_out_empty(time_set)
#     # Energy injected into the grid by the household
#     p_slack_in = slack.get_flow_in_empty(time_set)

#     battery = Battery()
#     # input electric power of the battery
#     p_bess_in = battery.get_injection_pulp_empty(time_set)
#     # output electric power of the battery
#     p_bess_out = battery.get_withdrawal_pulp_empty(time_set)
#     # stored electric energy
#     e_bess_stor = battery.get_stored_energy_pulp_empty(time_set)

#     time_set = range(time_set)

#     variables = {
#         "p_cons": p_cons,
#         "p_pv": p_pv,
#         "p_slack_out": p_slack_out,
#         "p_slack_in": p_slack_in,
#         "p_bess_in": p_bess_in,
#         "p_bess_out": p_bess_out,
#         "e_bess_stor": e_bess_stor,
#         "max_power_bess": max_power_bess,
#         "max_stored_energy_bess": max_stored_energy_bess,
#     }

#     return time_set, variables
