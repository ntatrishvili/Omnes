import numpy as np
import pulp

from app import model
from app.conversion.pulp_converter import PulpConverter
from app.infra.configuration import Config
from app.infra.logging_setup import get_logger
from app.infra.visualize import plot_energy_flows

log = get_logger(__name__)


def optimize_energy_system_pulp(model,**kwargs):
    log.info(f"Starting optimization with PuLP for {model.id} model")
    converter = PulpConverter()
    pulp_variables = converter.convert_model(model, **kwargs)
    time_set = pulp_variables["time_set"]

    # Automatically detect all components
    # TODO: improve understanding of component types, we need more automatic indicators
    pv_names = model.find_all_of_type_in_subentities("PV")
    load_names = model.find_all_of_type_in_subentities("Load")
    bess_names = model.find_all_of_type_in_subentities("Battery")
    slack_names = model.find_all_of_type_in_subentities("Slack")
    log.info(f"Detected PV entities: {pv_names}")
    log.info(f"Detected load entities: {load_names}")
    log.info(f"Detected batteries: {bess_names}")
    log.info(f"Detected slack entities: {slack_names}")

    prob = pulp.LpProblem("CSCopt", pulp.LpMinimize)

    # Global energy balance
    for t in range(time_set.number_of_time_steps):
        total_pv = np.sum(pulp_variables[f"{pv}.p_out"][t] for pv in pv_names)
        total_load = np.sum(pulp_variables[f"{load}.p_cons"][t] for load in load_names)
        total_bess_in = pulp.lpSum(pulp_variables[f"{b}.p_in"][t] for b in bess_names)
        total_bess_out = pulp.lpSum(pulp_variables[f"{b}.p_out"][t] for b in bess_names)
        total_slack_in = pulp.lpSum(pulp_variables[f"{s}.p_in"][t] for s in slack_names)
        total_slack_out = pulp.lpSum(pulp_variables[f"{s}.p_out"][t] for s in slack_names)

        # Energy balance constraint
        # We need to know which one is in and which one is out
        # Generation + Discharge + Import = Consumption + Charge + Export
        prob += (
            total_pv + total_slack_out + total_bess_out
            == total_load + total_slack_in + total_bess_in
        )

        # Battery constraints
        for b in bess_names:
            p_in = pulp_variables[f"{b}.p_in"]
            p_out = pulp_variables[f"{b}.p_out"]
            e_stor = pulp_variables[f"{b}.e_stor"]
            cap = pulp_variables[f"{b}.capacity"]
            max_p = pulp_variables[f"{b}.max_charge_rate"]

            # maximum input and output power and mutual exclusivity
            k = (t + 1) % time_set.number_of_time_steps
            prob += e_stor[k] - e_stor[t] == p_in[t] - p_out[t]
            prob += e_stor[t] <= cap
            prob += p_in[t] <= min(
                max_p, (total_pv - total_load if total_pv > total_load else 0)
            )
            prob += p_out[t] <= min(
                max_p, (total_load - total_pv if total_load > total_pv else 0)
            )

    # Objective: minimize total exchange with the grid
    prob += pulp.lpSum(
        pulp_variables[f"{s}.p_in"][t] + pulp_variables[f"{s}.p_out"][t]
        for s in slack_names
    )

    status = prob.solve()
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Optimization failed: {pulp.LpStatus[status]}")

    log.info(f"Optimization successful: {pulp.LpStatus[status]}")
    log.info(f"Objective value: {pulp.value(prob.objective)}")


    converter.convert_back(model, problem=prob, pulp_variables=pulp_variables, **kwargs)

    config = Config()
    plot_energy_flows(
        model,
        time_range_to_plot=time_set.time_points[136*24:137*24],
        output_path=config.get("path", "output"),
    )
    return {
        "status": pulp.LpStatus[status],
        "objective": pulp.value(prob.objective),
        "model": model,
        **kwargs
    }
