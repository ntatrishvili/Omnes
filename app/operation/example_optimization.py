import numpy as np
import pulp

from app.infra.configuration import Config
from app.infra.logging_setup import get_logger
from app.infra.visualize import plot_energy_flows

log = get_logger(__name__)


def optimize_energy_system(**kwargs):
    time_set = kwargs["time_set"]

    # Automatically detect all components
    # TODO: improve understanding of component types, we need more automatic indicators
    pv_names = {
        k.replace(".p_out", "")
        for k in kwargs
        if ("pv" in k.lower() or "sgen" in k.lower()) and ".p_out" in k
    }
    log.info(f"Detected PV entities: {pv_names}")
    load_names = {
        k.replace(".p_cons", "")
        for k in kwargs
        if "load" in k.lower() and ".p_cons" in k
    }
    log.info(f"Detected load entities: {load_names}")
    bess_names = {
        k.replace(".p_in", "")
        for k in kwargs
        if "battery" in k.lower() and ".p_in" in k
    }
    log.info(f"Detected batteries: {bess_names}")
    slack_names = {
        k.replace(".p_in", "")
        for k in kwargs
        if ("slack" in k.lower() or "grid" in k.lower()) and ".p_in" in k
    }
    log.info(f"Detected slack entities: {slack_names}")

    prob = pulp.LpProblem("CSCopt", pulp.LpMinimize)

    # Global energy balance
    for t in time_set:
        total_pv = np.sum(kwargs[f"{pv}.p_out"][t] for pv in pv_names)
        total_load = np.sum(kwargs[f"{load}.p_cons"][t] for load in load_names)
        total_bess_in = pulp.lpSum(kwargs[f"{b}.p_in"][t] for b in bess_names)
        total_bess_out = pulp.lpSum(kwargs[f"{b}.p_out"][t] for b in bess_names)
        total_slack_in = pulp.lpSum(kwargs[f"{s}.p_in"][t] for s in slack_names)
        total_slack_out = pulp.lpSum(kwargs[f"{s}.p_out"][t] for s in slack_names)

        # Energy balance constraint
        # We need to know which one is in and which one is out
        # Generation + Discharge + Import = Consumption + Charge + Export
        prob += (
            total_pv + total_slack_out + total_bess_out
            == total_load + total_slack_in + total_bess_in
        )

        # Battery constraints
        for b in bess_names:
            p_in = kwargs[f"{b}.p_in"]
            p_out = kwargs[f"{b}.p_out"]
            e_stor = kwargs[f"{b}.e_stor"]
            cap = kwargs[f"{b}.capacity"]
            max_p = kwargs[f"{b}.max_charge_rate"]

            # maximum input and output power and mutual exclusivity
            k = (t + 1) % len(time_set)
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
        kwargs[f"{s}.p_in"][t] + kwargs[f"{s}.p_out"][t]
        for s in slack_names
        for t in time_set
    )

    status = prob.solve(pulp.GUROBI_CMD(msg=True))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Optimization failed: {pulp.LpStatus[status]}")

    days = [136, 137]
    config = Config()
    plot_energy_flows(
        kwargs,
        pv_names,
        load_names,
        bess_names,
        slack_names,
        time_range_to_plot=range(24 * days[0], 24 * days[1]),
        output_path=config.get("path", "output"),
    )

    for b in bess_names:
        for name in ["p_in", "p_out", "e_stor"]:
            kwargs[f"{b}.{name}"] = np.array(
                [pulp.value(kwargs[f"{b}.{name}"][t]) for t in time_set]
            )

    for s in slack_names:
        for name in ["p_in", "p_out"]:
            kwargs[f"{s}.{name}"] = np.array(
                [pulp.value(kwargs[f"{s}.{name}"][t]) for t in time_set]
            )

    return {
        "status": pulp.LpStatus[status],
        "objective": pulp.value(prob.objective),
        **kwargs,
    }
