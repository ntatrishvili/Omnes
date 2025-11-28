from os.path import join

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pulp
from pandas import Timestamp, Timedelta

from utils.configuration import Config
from utils.logging_setup import get_logger

log = get_logger(__name__)

import matplotlib as mpl

mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["figure.facecolor"] = "none"
mpl.rcParams["axes.facecolor"] = "none"


def plot_energy_flows(
    kwargs,
    pv_names,
    load_names,
    bess_names,
    slack_names,
    time_range_to_plot=None,
    output_path=None,
):
    time_set = kwargs["time_set"]
    nT = len(time_set)
    if time_range_to_plot is None:
        time_range_to_plot = range(nT)

    ax, twin_ax = None, None
    # --- Extract and sum all variables ---
    # PV production
    pv_profiles = {
        pv: np.array([pulp.value(kwargs[f"{pv}.p_out"][t]) for t in time_set])
        for pv in pv_names
    }
    pv_sum = sum(pv_profiles.values())
    print("PV total production:", pv_sum.max())

    # Loads
    load_profiles = {
        ld: np.array([pulp.value(kwargs[f"{ld}.p_cons"][t]) for t in time_set])
        for ld in load_names
    }
    load_sum = sum(load_profiles.values())
    print("Total load:", load_sum.max())

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
    bottom_pv = np.zeros_like(time_set, dtype=float)
    for color, (pv, p) in zip(colors_pv, pv_profiles.items()):
        plt.bar(
            time_range_to_plot,
            -p[time_range_to_plot],
            color=color,
            width=1.0,
            label=f"{pv.replace('LV1.101 ', '')} (PV)",
            bottom=bottom_pv[time_range_to_plot],
        )
        bottom_pv[time_range_to_plot] -= p[time_range_to_plot]
    plt.plot(
        time_range_to_plot, -pv_sum[time_range_to_plot], "r", lw=2.5, label="Total PV"
    )

    # --- Plot loads ---
    bottom_loads = np.zeros_like(time_set, dtype=float)
    for color, (ld, p) in zip(colors_load, load_profiles.items()):
        plt.bar(
            time_range_to_plot,
            p[time_range_to_plot],
            color=color,
            width=1.0,
            label=f"{ld.replace('LV1.101 ', '')}",
            bottom=bottom_loads[time_range_to_plot],
        )
        bottom_loads[time_range_to_plot] += p[time_range_to_plot]

    plt.plot(
        time_range_to_plot,
        load_sum[time_range_to_plot],
        "--",
        lw=2.8,
        label="Total Load",
        color=colors_load[0],
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
            edgecolor="darkgrey",
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
            edgecolor="darkgrey",
            bottom=bottom_discharge[time_range_to_plot],
        )
        bottom_discharge[time_range_to_plot] -= p_out[time_range_to_plot]

        # --- Slack ---
        slack_p = slack_in_sum - slack_out_sum
        slack_in = slack_p.copy()
        slack_in[slack_in > 0] = 0
        slack_out = slack_p.copy()
        slack_out[slack_out < 0] = 0
        plt.plot(
            time_range_to_plot,
            slack_in[time_range_to_plot],
            "k:",
            lw=1.5,
            label="Slack export",
        )
        plt.plot(
            time_range_to_plot,
            slack_out[time_range_to_plot],
            "k--",
            lw=1.5,
            label="Slack import",
        )

    # --- Plot stored energy (averaged scaling) ---
    if e_bess_profiles:
        ax = plt.gca()
        twin_ax = ax.twinx()
        color = "b"

        e_sum = sum(e_bess_profiles.values())
        twin_ax.step(
            time_range_to_plot,
            e_sum[time_range_to_plot],
            "b",
            alpha=0.8,
            lw=2,
            where="mid",
            label="Stored energy",
        )
        # color the y-axis (ticks, label, spine) to match the step color
        twin_ax.set_ylabel("Stored energy [kWh]", color=color)
        twin_ax.yaxis.label.set_color(color)
        twin_ax.tick_params(axis="y", colors=color)
        twin_ax.spines["right"].set_color(color)
        ax.set_ylabel("Power [kWh]")

    # plt.ylabel("Energy flows (kWh)")
    plt.xlabel("Time (quarter-hours)")
    plt.title("Energy System Operation Overview")
    # merge legends from main axis and twin axis
    if ax or twin_ax:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = twin_ax.get_legend_handles_labels()
        if handles1 or handles2:
            ax.legend(
                handles1 + handles2, labels1 + labels2, loc="best", fontsize=9, ncol=2
            )
    else:
        plt.legend(loc="best", fontsize=9, ncol=2)

    x_hours = np.array(list(time_range_to_plot), dtype=int)
    # choose base date (year can be parameterized); ordinal hours count from Jan 1 00:00
    base_year = 2016
    base = Timestamp(year=base_year, month=1, day=1)
    # choose tick spacing depending on total span
    span_hours = x_hours.max() - x_hours.min() if x_hours.size else 0
    if span_hours <= 24:
        tick_step = 2
    elif span_hours <= 72:
        tick_step = 3
    elif span_hours <= 24 * 14:
        tick_step = 6
    else:
        tick_step = 24

    # pick tick positions from the numeric x-axis (matching plotted x positions)
    tick_idx = np.arange(x_hours.min(), x_hours.max() + 1, tick_step, dtype=int)

    # build labels from base date + hours
    tick_labels = [(base + Timedelta(hours=int(h))).strftime("%H:00") for h in tick_idx]

    # set ticks and labels
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    # add the date as x-axis label (use the day of the first plotted hour)
    start_date = (base + Timedelta(hours=int(x_hours.min()))).date()
    ax.set_ylabel(f"Power [kW]")
    ax.set_xlabel(f"Time (hours) — date: {start_date:%Y-%m-%d}")
    plt.tight_layout()
    plt.savefig(join(output_path, "energy_system_operation.png"))


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
