"""Named EV-charging environments and reward-weight profiles of the RL showcase.

The same campus (9 chargers, 2009 kWp PV) is run in different **environments** (how hard it
is to serve the vehicles, whether the battery hides the EV timing) and with different
**profiles** (how the payoff weighs PV use against line overloading and unmet energy). The
learned policies of the profiles are compared with the rule-based baselines.

Sessions come from the measured campus mixtures (``gmm_events.py``).

Environments
* ``standard``    workplace parking: vehicles stay for hours, every session can be served;
                  no battery, so only the EV timing decides what the PV surplus does
* ``congested``   short stays and a connection limit per bus: not every session can be served,
                  so the unmet-energy weight has to be traded against the other goals
* ``with_battery`` ``standard`` plus the NaS battery of the campus (it soaks up the PV surplus
                  and hides the EV timing: the contrast case)
* ``hard``        ``with_battery`` plus an operating limit of 250 kVA on transformer 4T (where
                  the battery and the b4 / b26 roofs and chargers are): the rule-based
                  policies know neither the battery nor the transformer

Profiles (weights of the payoff terms; voltage / transformer terms never bind and are kept
as safety terms)
* ``pv_max``       use the PV surplus: import and export weigh 1, line overload 0
* ``grid_safe``    relieve the overloaded feeders: line overload weighs most
* ``balanced``     both at once
* ``comfort_first`` like ``balanced`` but unmet energy weighs much more
"""

import copy
from typing import Optional

from app.operation.rl.problem import (
    EventSpec,
    ObservationTerm,
    RewardTerm,
    RLProblem,
    default_observation_terms,
)

# generator parameters of ``GmmBundleGenerator`` (gmm_events.py): the measured mixtures of
# start, energy and stay of the campus; only the plug-in rate is set here: attempts per charger
# and day (an attempt during a stay in progress is dropped, so about 1.5 sessions per charger
# and working day are served at 4 attempts; 8 attempts give about 2.2 but not larger relative
# effects)
EVENTS = {
    "standard": dict(workday_rate=4.0, weekend_rate=1.6),
    "congested": dict(workday_rate=4.0, weekend_rate=1.6),
}

# ``safety``: the laxity layer forces the minimum power that still delivers the energy, so
# the agent only learns *when* to charge; without it (congested) it also decides *who* is served
# ``profiles``: the profiles trained by default; ``comfort_first`` only differs from ``balanced``
# in the weight of the unmet energy, which is irrelevant where the laxity layer delivers it
# ``cap_kw``: connection limit of all chargers on one bus; 7 kW leaves about 9 % of the
# requested energy unserved at 4 plug-ins per charger and day
ENVIRONMENTS = {
    "standard": dict(
        events="standard",
        battery="none",
        cap_kw=None,
        clip=True,
        safety="laxity",
        profiles=("pv_max", "grid_safe", "balanced"),
    ),
    "congested": dict(
        events="congested",
        battery="none",
        cap_kw=7.0,
        clip=False,
        safety="none",
        profiles=("pv_max", "grid_safe", "balanced", "comfort_first"),
    ),
    "with_battery": dict(
        events="standard",
        battery="self_consumption_rule",
        cap_kw=None,
        clip=True,
        safety="laxity",
        profiles=("pv_max", "grid_safe", "balanced"),
    ),
    # the harder case: the agent also dispatches the NaS battery (the rules leave it to the
    # self-consumption rule, which is blind to the local grid). The battery sits on transformer
    # 4T, which has an operating limit (connection agreement) that the b4 / b26 roofs exceed
    # on about half of the days; only the battery (250 kW) can relieve it, the chargers (110 kW
    # on 4T) cannot
    "hard": dict(
        events="standard",
        battery="agent",
        cap_kw=None,
        clip=True,
        safety="laxity",
        profiles=("pv_max", "grid_safe", "balanced"),
        trafo_limits={"TR_4T": 250.0},
    ),
}

# weight per reward metric (cost per kWh / kVAh)
# (``trafo_overload_kvah`` only counts where an environment sets a transformer limit)
PROFILES = {
    "pv_max": dict(
        import_kwh=1.0,
        export_kwh=1.0,
        unmet_kwh=5.0,
        line_overload_kvah=0.0,
        trafo_overload_kvah=0.0,
    ),
    "grid_safe": dict(
        import_kwh=0.1,
        export_kwh=0.1,
        unmet_kwh=5.0,
        line_overload_kvah=10.0,
        trafo_overload_kvah=10.0,
    ),
    "balanced": dict(
        import_kwh=0.5,
        export_kwh=0.5,
        unmet_kwh=5.0,
        line_overload_kvah=5.0,
        trafo_overload_kvah=5.0,
    ),
    "comfort_first": dict(
        import_kwh=0.5,
        export_kwh=0.5,
        unmet_kwh=25.0,
        line_overload_kvah=5.0,
        trafo_overload_kvah=5.0,
    ),
}
SAFETY = dict(voltage_excess_pu=200.0)


def default_problem() -> RLProblem:
    """The campus problem the environments are made from (``make_problem`` sets the rest).

    2009 kWp PV (9 existing + 2000 added), 1450 kWh NaS battery, 9 chargers; the observation
    adds the line loading, the local PV minus load of every charger's bus and the urgency
    rank / occupancy of the vehicles sharing a connection.
    """
    return RLProblem(
        id="kfki_ev_v1g",
        scenario=dict(added_pv_kwp=2000.0, battery_kwh=1450.0, n_chargers=9),
        events=EventSpec(source="generator", generator="gmm_bundle"),
        observation=default_observation_terms()
        + [
            ObservationTerm("charger_line_loading_prev"),
            ObservationTerm("charger_site_injection"),
            ObservationTerm("charger_site_rank"),
            ObservationTerm("charger_site_count"),
        ],
    )


def reward_terms(weights: dict[str, float]) -> list[RewardTerm]:
    """Payoff terms of a profile (plus the safety terms)."""
    terms = [RewardTerm(m, w) for m, w in {**weights, **SAFETY}.items()]
    for t in terms:
        if t.metric == "voltage_excess_pu":
            t.params = {"v_hi": 1.05, "v_lo": 0.95}
    return terms


def make_problem(
    base: RLProblem, environment: str = "standard", profile: Optional[str] = "balanced"
) -> RLProblem:
    """Copy of ``base`` set to an environment and a reward profile."""
    env = ENVIRONMENTS[environment]
    p = copy.deepcopy(base)
    p.battery_policy = env["battery"]
    p.power_flow_mode = "feeder"  # fast: line overload of the charger feeders in closed form
    p.events.source = "generator"
    p.events.generator = "gmm_bundle"
    p.events.generator_params = dict(EVENTS[env["events"]])
    p.events.clip_to_deliverable = env["clip"]
    p.action.site_power_cap_kw = env["cap_kw"]
    p.action.safety_layer = env["safety"]
    p.trafo_limits_kva = dict(env.get("trafo_limits", {}))
    # the cost of a charging step shows up in the same step (price of the moment), while
    # the random arrivals of later sessions are noise: a short horizon learns much faster
    p.agent.gamma = 0.8
    p.agent.total_timesteps = 150_000
    if profile is not None:
        p.reward = reward_terms(PROFILES[profile])
    p.id = f"{base.id}_{environment}" + (f"_{profile}" if profile else "")
    p.validate()
    return p
