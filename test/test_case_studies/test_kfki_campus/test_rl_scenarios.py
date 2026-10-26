"""RL showcase: site connection limit, feeder loading, line / laxity baselines, profiles."""

import numpy as np
import pandas as pd
import pytest

from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.rl.baselines import (
    LeastLaxityAgent,
    LineFollowingAgent,
    UncontrolledAgent,
)
from app.case_studies.kfki_campus.rl.env import KfkiChargingEnv
from app.case_studies.kfki_campus.rl.evaluate import evaluate_policies
from app.case_studies.kfki_campus.rl.profiles import (
    ENVIRONMENTS,
    PROFILES,
    make_problem,
)
from app.case_studies.kfki_campus.synthetic import make_synthetic_csv
from app.operation.rl.events import ChargingEvent
from app.operation.rl.observations import observation_size
from app.operation.rl.problem import (
    EpisodeSpec,
    EventSpec,
    ObservationTerm,
    RLProblem,
    default_observation_terms,
)
from app.operation.rl.rollout import run_episode

START, END = "2025-06-02 00:00", "2025-06-05 00:00"
B4 = ["ev_c04", "ev_c05"]  # chargers 4 and 5 share bus 4T_b4 (assets.xlsx:ev_sites)


@pytest.fixture(scope="module")
def data():
    return load_campus_data()


@pytest.fixture(scope="module")
def csv(tmp_path_factory, data):
    return make_synthetic_csv(
        tmp_path_factory.mktemp("rls") / "ts.csv", start="2025-06-02", days=3, data=data
    )


def _problem(**kw):
    base = dict(
        scenario=dict(n_chargers=5, added_pv_kwp=2000.0, battery_kwh=0.0),
        power_flow_mode="post",
        battery_policy="none",
        episode=EpisodeSpec(train_fraction=0.67),
        events=EventSpec(source="file", session_file="x"),
        observation=default_observation_terms()
        + [
            ObservationTerm("charger_line_loading_prev"),
            ObservationTerm("charger_site_injection"),
        ],
    )
    base.update(kw)
    return RLProblem(**base)


def _env(csv, data, problem, events):
    return KfkiChargingEnv(problem, csv, data, START, END, events, "train")


def _event(cid, hour, dwell_h, kwh):
    t = pd.Timestamp(START) + pd.Timedelta(hours=hour)
    return ChargingEvent(f"s_{cid}", cid, t, dwell_h, kwh)


def _run(env, agent, steps=24):
    agent.bind(env)
    obs, _ = env.reset(seed=0, options={"day": 0})
    for _ in range(steps):
        obs, *_ = env.step(agent.predict(obs))


def test_chargers_five_and_four_share_a_bus_on_the_overloaded_line(csv, data):
    env = _env(csv, data, _problem(), [])
    assert len(env._groups) == 4  # b26, b29, b17 and b4 (two chargers)
    g = next(g for g in env._groups if len(g) == 2)
    assert [list(env.chargers)[i] for i in g] == B4
    assert env.net.line.name[env._ev_line[g[0]]] == "L_4T_b4"


def test_site_cap_limits_the_chargers_of_one_bus(csv, data):
    p = _problem()
    p.action.site_power_cap_kw = 22.0
    env = _env(csv, data, p, [_event(c, 8, 4, 44.0) for c in B4])
    _run(env, UncontrolledAgent(p))
    ev = env.ev_power()
    assert (ev[B4].sum(axis=1) <= 22.0 + 1e-9).all()
    assert np.allclose(ev["ev_c04"], ev["ev_c05"])  # proportional split
    assert env.sessions()["delivered_kwh"].sum() == pytest.approx(88.0)


def test_requests_that_cannot_be_served_end_with_unmet_energy(csv, data):
    ev = [_event("ev_c04", 8, 2, 100.0)]  # 100 kWh in 2 h at 22 kW
    for clip, expect_unmet in ((True, 0.0), (False, 56.0)):
        p = _problem()
        p.events.clip_to_deliverable = clip
        env = _env(csv, data, p, ev)
        _run(env, UncontrolledAgent(p))
        s = env.sessions()
        assert (s["requested_kwh"] - s["delivered_kwh"]).sum() == pytest.approx(
            expect_unmet
        )


def test_least_laxity_serves_the_urgent_vehicle_under_the_cap(csv, data):
    p = _problem()
    p.action.site_power_cap_kw = 22.0
    events = [_event("ev_c04", 8, 1, 22.0), _event("ev_c05", 8, 4, 22.0)]
    unmet = {}
    for name, agent in (
        ("uncontrolled", UncontrolledAgent(p)),
        ("least_laxity", LeastLaxityAgent(p)),
    ):
        env = _env(csv, data, p, events)
        _run(env, agent)
        s = env.sessions().set_index("charger_id")
        unmet[name] = float((s["requested_kwh"] - s["delivered_kwh"]).sum())
    assert unmet["uncontrolled"] == pytest.approx(11.0)  # the urgent car got half
    assert unmet["least_laxity"] == pytest.approx(0.0)  # urgent first, the other later


def test_line_following_absorbs_the_export_of_its_feeder(csv, data):
    p = _problem()
    env = _env(csv, data, p, [_event(c, 8, 8, 50.0) for c in B4])
    agent = LineFollowingAgent(p)
    agent.bind(env)
    env.reset(seed=0, options={"day": 0})
    for _ in range(9):  # the vehicles are connected from step 8
        env.step(np.zeros(5))
    t = env.t0 + env.k
    env._site_inj[t] = 0.0
    quiet = agent.predict(None)
    env._site_inj[t] = 1e4  # feeder far above its rating
    loud = agent.predict(None)
    assert quiet[3] < 0.5 and loud[3] == 1.0 and loud[4] == 1.0
    assert loud[0] == 0.0  # no vehicle at charger 1


def test_observation_has_the_line_terms(csv, data):
    p = _problem()
    env = _env(csv, data, p, [])
    obs, _ = env.reset(seed=0, options={"day": 0})
    assert obs.shape == (observation_size(p.observation, 5),)
    s = env._observe_state()
    assert s.site_injection_kw.shape == (5,) and np.isfinite(s.line_sn_kva).all()


def test_feeder_mode_matches_the_power_flow(csv, data):
    p = _problem(power_flow_mode="feeder")
    events = [_event(c, 8, 8, 60.0) for c in B4 + ["ev_c01"]]
    env = _env(csv, data, p, events)
    _run(env, UncontrolledAgent(p))
    lines = np.unique(env._ev_line[env._ev_line >= 0])
    fast = env.history("line")[:, lines]
    exact = env.sim_result().line_loading_pct[:, lines]  # post-hoc power flow
    assert np.allclose(fast, exact, atol=4.0)  # percentage points


def test_profiles_and_environments_build_valid_problems():
    base = RLProblem(scenario=dict(n_chargers=9))
    for env in ENVIRONMENTS:
        for prof, weights in PROFILES.items():
            p = make_problem(base, env, prof)
            w = p.reward_weights()
            assert all(w[m] == v for m, v in weights.items())
            assert p.power_flow_mode == "feeder"
    assert make_problem(base, "congested", None).action.site_power_cap_kw == 7.0
    assert make_problem(base, "standard", None).events.generator == "gmm_bundle"
    assert all(ENVIRONMENTS[e]["profiles"] for e in ENVIRONMENTS)
    assert make_problem(base, "standard", None).action.safety_layer == "laxity"
    assert make_problem(base, "congested", None).events.clip_to_deliverable is False
    assert make_problem(base, "with_battery", None).battery_policy != "none"
    assert base.action.site_power_cap_kw is None  # the base problem is not modified


def test_evaluation_scores_every_policy_with_every_profile(csv, data):
    p = _problem()
    events = [_event(c, 8, 6, 30.0) for c in B4]
    env = _env(csv, data, p, events)
    env.split = "test"
    ev = evaluate_policies(
        env,
        {"uncontrolled": UncontrolledAgent(p), "line_following": LineFollowingAgent(p)},
        days=[0],
        seed=0,
        keep_episodes=False,
    )
    k = ev.kpis.set_index("policy")
    for prof in PROFILES:
        assert k.loc["uncontrolled", f"gain_{prof}"] == pytest.approx(0.0)
    assert k.loc["uncontrolled", "reward_vs_uncontrolled"] == 0.0
    assert {"ev_line_overload_kvah", "ev_grid_kwh", "line_overload_hours"} <= set(k)
