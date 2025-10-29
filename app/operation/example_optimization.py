import time

import numpy as np
import pulp
from pulp import LpStatusOptimal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.logging_setup import get_logger

log = get_logger(__name__)


def plot_energy_flows(kwargs, time_range_to_plot=None):
    time_set = kwargs["time_set"]
    nT = len(time_set)
    if time_range_to_plot is None:
        time_range_to_plot = range(nT)

    # Identify all components automatically
    pv_names = {k.split(".")[0] for k in kwargs if k.startswith("pv") and ".p_out" in k}
    load_names = {
        k.split(".")[0] for k in kwargs if k.startswith("load") and ".p_cons" in k
    }
    bess_names = {
        k.split(".")[0] for k in kwargs if k.startswith("battery") and ".p_in" in k
    }
    slack_names = {
        k.split(".")[0] for k in kwargs if k.startswith("slack") and ".p_in" in k
    }

    # --- Extract and sum all variables ---
    # PV production
    pv_profiles = {
        pv: np.array([pulp.value(kwargs[f"{pv}.p_out"][t]) for t in time_set])
        for pv in pv_names
    }
    pv_sum = sum(pv_profiles.values())

    # Loads
    load_profiles = {
        ld: np.array([pulp.value(kwargs[f"{ld}.p_cons"][t]) for t in time_set])
        for ld in load_names
    }
    load_sum = sum(load_profiles.values())

    # Battery flows
    bess_in_profiles = {
        b: np.array([pulp.value(kwargs[f"{b}.p_in"][t]) for t in time_set])
        for b in bess_names
    }
    bess_out_profiles = {
        b: np.array([pulp.value(kwargs[f"{b}.p_out"][t]) for t in time_set])
        for b in bess_names
    }
    e_bess_profiles = {
        b: np.array([pulp.value(kwargs[f"{b}.e_stor"][t]) for t in time_set])
        for b in bess_names
    }
    bess_in_sum = sum(bess_in_profiles.values()) if bess_in_profiles else np.zeros(nT)
    bess_out_sum = (
        sum(bess_out_profiles.values()) if bess_out_profiles else np.zeros(nT)
    )

    # Slack (import/export)
    slack_in_sum = np.sum(
        [
            np.array([pulp.value(kwargs[f"{s}.p_in"][t]) for t in time_set])
            for s in slack_names
        ],
        axis=0,
    )
    slack_out_sum = np.sum(
        [
            np.array([pulp.value(kwargs[f"{s}.p_out"][t]) for t in time_set])
            for s in slack_names
        ],
        axis=0,
    )

    # --- Compute basic metrics ---
    total_charge_hours = sum(np.sum(v > 0) for v in bess_in_profiles.values()) // 4
    total_discharge_hours = sum(np.sum(v > 0) for v in bess_out_profiles.values()) // 4
    print(f"Total charge hours: {total_charge_hours}")
    print(f"Total discharge hours: {total_discharge_hours}")

    # --- Plot setup ---
    plt.figure(figsize=(10, 6))
    colors_bess = cm.Greys(np.linspace(0.4, 0.8, len(bess_names))) if bess_names else []
    colors_pv = cm.Reds(np.linspace(0.5, 0.9, len(pv_names))) if pv_names else []
    colors_load = (
        cm.Greens(np.linspace(0.5, 0.9, len(load_names))) if load_names else []
    )

    # --- Plot PVs ---
    for color, (pv, p) in zip(colors_pv, pv_profiles.items()):
        plt.plot(
            time_range_to_plot,
            p[time_range_to_plot],
            color=color,
            lw=1.5,
            label=f"{pv} (PV)",
        )
    plt.plot(
        time_range_to_plot, pv_sum[time_range_to_plot], "r", lw=2.5, label="Total PV"
    )

    # --- Plot loads ---
    for color, (ld, p) in zip(colors_load, load_profiles.items()):
        plt.plot(
            time_range_to_plot,
            p[time_range_to_plot],
            color=color,
            lw=1.5,
            linestyle="--",
            label=f"{ld} (Load)",
        )
    plt.plot(
        time_range_to_plot,
        load_sum[time_range_to_plot],
        "g--",
        lw=2.5,
        label="Total Load",
    )

    # --- Plot battery charge/discharge ---
    bottom_charge = np.zeros_like(time_set, dtype=float)
    bottom_discharge = np.zeros_like(time_set, dtype=float)
    for color, (b, p_in) in zip(colors_bess, bess_in_profiles.items()):
        plt.bar(
            time_range_to_plot,
            p_in[time_range_to_plot],
            color=color,
            width=1.0,
            label=f"{b} charge",
            bottom=bottom_charge[time_range_to_plot],
        )
        bottom_charge[time_range_to_plot] += p_in[time_range_to_plot]
    for color, (b, p_out) in zip(colors_bess, bess_out_profiles.items()):
        plt.bar(
            time_range_to_plot,
            -p_out[time_range_to_plot],
            color=color,
            width=1.0,
            alpha=0.7,
            label=f"{b} discharge",
            bottom=bottom_discharge[time_range_to_plot],
        )
        bottom_discharge[time_range_to_plot] -= p_out[time_range_to_plot]

    # --- Plot stored energy (averaged scaling) ---
    if e_bess_profiles:
        e_sum = sum(e_bess_profiles.values())
        plt.step(
            time_range_to_plot,
            e_sum[time_range_to_plot] / 8,
            "b",
            alpha=0.8,
            lw=2,
            label="Stored energy / 8",
        )

    # --- Slack ---
    plt.plot(
        time_range_to_plot,
        slack_in_sum[time_range_to_plot],
        "k:",
        lw=1.5,
        label="Slack import",
    )
    plt.plot(
        time_range_to_plot,
        -slack_out_sum[time_range_to_plot],
        "k--",
        lw=1.5,
        label="Slack export",
    )

    plt.ylabel("Power / Energy (kWh)")
    plt.xlabel("Time (quarter-hours)")
    plt.title("Energy System Operation Overview")
    plt.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.show()


def optimize_energy_system(**kwargs):
    time_set = kwargs["time_set"]

    # Automatically detect all components
    pv_names = {k.split(".")[0] for k in kwargs if k.startswith("pv") and ".p_out" in k}
    load_names = {
        k.split(".")[0] for k in kwargs if k.startswith("load") and ".p_cons" in k
    }
    bess_names = {
        k.split(".")[0] for k in kwargs if k.startswith("battery") and ".p_in" in k
    }
    slack_names = {
        k.split(".")[0] for k in kwargs if k.startswith("slack") and ".p_in" in k
    }

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

    plot_energy_flows(kwargs, time_range_to_plot=range(11040, 11136))

    return {
        "status": pulp.LpStatus[status],
        "objective": pulp.value(prob.objective),
    }
