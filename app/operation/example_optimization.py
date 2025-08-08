import time

import numpy as np
import pulp
from pulp import LpStatusOptimal


def optimize(**kwargs) -> None:
    time_set = kwargs["time_set"]
    p_cons = kwargs["p_cons"]
    p_pv = kwargs["p_pv"]
    p_slack_out = kwargs["p_slack_out"]
    p_slack_in = kwargs["p_slack_in"]
    p_bess_in = kwargs["p_bess_in"]
    p_bess_out = kwargs["p_bess_out"]
    e_bess_stor = kwargs["e_bess_stor"]
    max_power_bess = kwargs["max_charge_rate"]
    max_stored_energy_bess = kwargs["capacity"]
    prob = pulp.LpProblem("CSCopt", pulp.LpMinimize)

    # Add the constraints to the problem
    for t in time_set:
        # subsequent time-step (0 for the last one, i.e., cyclic)
        k = (t + 1) % len(time_set)

        # Energy balance between the slack and the household
        prob += (
            p_pv[t] + p_slack_out[t] + p_bess_out[t]
            == p_slack_in[t] + p_bess_in[t] + p_cons[t]
        )

        # Battery energy storage system
        # energy balance between time steps
        prob += e_bess_stor[k] - e_bess_stor[t] == p_bess_in[t] - p_bess_out[t]

        # maximum input and output power and mutual exclusivity
        prob += p_bess_in[t] <= min(
            max_power_bess,
            (p_pv[t] - p_cons[t] if p_pv[t] > p_cons[t] else 0),
        )
        prob += p_bess_out[t] <= min(
            max_power_bess,
            (p_cons[t] - p_pv[t] if p_cons[t] > p_pv[t] else 0),
        )

        # maximum storable energy (minimum is defined by variable lower bound)
        prob += e_bess_stor[t] <= max_stored_energy_bess

    # Objective
    prob += pulp.lpSum([p_slack_out[t] + p_slack_in[t] for t in time_set])

    # Solve the problem
    t = time.time()
    status = prob.solve(pulp.GUROBI_CMD(msg=True))
    if status != LpStatusOptimal:
        raise RuntimeError("Unable to solve the problem!")
    print(f"Time to solve: {time.time() - t:.3}")
    objective = pulp.value(prob.objective)

    print(f"Optimization Status: {prob.status}")
    print(f"Energy exchanged with the slack for the entire year: {objective:.0}")

    # Retain variables
    p_bess_in = np.array([pulp.value(p_bess_in[t]) for t in time_set])
    p_bess_out = np.array([pulp.value(p_bess_out[t]) for t in time_set])
    e_bess_stor = np.array([pulp.value(e_bess_stor[t]) for t in time_set])

    charge_hours = np.sum(p_bess_in > 0) // 4
    discharge_hours = np.sum(p_bess_out > 0) // 4
    print(f"Charge hours: {charge_hours}")
    print(f"Discharge hours: {discharge_hours}")

    import matplotlib.pyplot as plt

    time_range_to_plot = range(11040, 11136)
    plt.bar(
        time_range_to_plot,
        p_bess_in[time_range_to_plot],
        color="grey",
        label="Battery charging",
    )

    plt.bar(
        time_range_to_plot,
        -p_bess_out[time_range_to_plot],
        color="darkgrey",
        label="Battery discharging",
    )

    plt.step(
        time_range_to_plot,
        e_bess_stor[time_range_to_plot] / 8,
        "b",
        alpha=0.7,
        label="Stored energy/8",
    )

    plt.plot(
        time_range_to_plot,
        p_pv[time_range_to_plot],
        "r",
        label="PV production",
    )
    plt.plot(
        time_range_to_plot,
        p_cons[time_range_to_plot],
        "g",
        label="Consumption",
    )
    plt.ylabel("Energy (kWh)")
    plt.xlabel("Time (quarter hours)")
    plt.legend()
    plt.tight_layout()
    plt.show()
