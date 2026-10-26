"""Oracle with the payoff weights, transformer limit, agent-controlled battery, warm start."""

import numpy as np
import pandas as pd
import pytest

from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.rl.baselines import (
    GridGuardAgent,
    LeastLaxityAgent,
    LineFollowingAgent,
    OracleAgent,
    PVFollowingAgent,
    UncontrolledAgent,
)
from app.case_studies.kfki_campus.rl.env import KfkiChargingEnv
from app.case_studies.kfki_campus.rl.evaluate import evaluate_policies
from app.case_studies.kfki_campus.synthetic import make_synthetic_csv
from app.operation.rl.events import ChargingEvent
from app.operation.rl.observations import OBSERVATION_REGISTRY
from app.operation.rl.problem import (
    AgentSpec,
    EpisodeSpec,
    EventSpec,
    RewardTerm,
    RLProblem,
)
from app.operation.rl.rollout import run_episode

START, END = "2025-06-02 00:00", "2025-06-05 00:00"
B4 = ["ev_c04", "ev_c05"]


@pytest.fixture(scope="module")
def data():
    return load_campus_data()


@pytest.fixture(scope="module")
def csv(tmp_path_factory, data):
    return make_synthetic_csv(
        tmp_path_factory.mktemp("ub") / "ts.csv", start="2025-06-02", days=3, data=data
    )


def _problem(**kw):
    base = dict(
        scenario=dict(n_chargers=5, added_pv_kwp=2000.0, battery_kwh=0.0),
        power_flow_mode="feeder",
        battery_policy="none",
        episode=EpisodeSpec(train_fraction=0.67),
        events=EventSpec(source="file", session_file="x"),
        reward=[
            RewardTerm("import_kwh", 1.0),
            RewardTerm("export_kwh", 1.0),
            RewardTerm("unmet_kwh", 5.0),
        ],
    )
    base.update(kw)
    return RLProblem(**base)


def _env(csv, data, problem, events):
    return KfkiChargingEnv(problem, csv, data, START, END, events, "test")


def _ev(cid, hour, dwell_h, kwh):
    t = pd.Timestamp(START) + pd.Timedelta(hours=hour)
    return ChargingEvent(f"s_{cid}_{hour}", cid, t, dwell_h, kwh)


def _events():
    return [_ev(c, h, 6, 30.0) for c in B4 + ["ev_c01"] for h in (7,)]


def test_oracle_is_not_worse_than_any_rule_under_a_cap(csv, data):
    p = _problem()
    p.action.site_power_cap_kw = 11.0
    p.events.clip_to_deliverable = False
    env = _env(csv, data, p, [_ev(c, 8, 3, 33.0) for c in B4])
    agents = {
        "uncontrolled": UncontrolledAgent(p),
        "pv_following": PVFollowingAgent(p),
        "least_laxity": LeastLaxityAgent(p),
        "oracle": OracleAgent(p),
    }
    k = evaluate_policies(env, agents, days=[0], seed=0, keep_episodes=False).kpis
    r = k.set_index("policy")["reward"]
    assert r["oracle"] >= r.drop("oracle").max() - 1e-6
    u = k.set_index("policy")["ev_unmet_kwh"]
    assert u["oracle"] <= u.drop("oracle").min() + 1e-6


def test_oracle_with_a_line_weight_delivers_the_energy(csv, data):
    p = _problem()
    p.reward.append(RewardTerm("line_overload_kvah", 10.0))
    env = _env(csv, data, p, _events())
    k = evaluate_policies(
        env, {"oracle": OracleAgent(p)}, days=[0], seed=0, keep_episodes=False
    ).kpis
    assert k["ev_unmet_kwh"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_transformer_limit_enters_reward_and_kpi(csv, data):
    p = _problem(trafo_limits_kva={"TR_4T": 5.0})  # absurdly low: always overloaded
    p.reward.append(RewardTerm("trafo_overload_kvah", 1.0))
    env = _env(csv, data, p, _events())
    ag = UncontrolledAgent(p)
    res = run_episode(env, ag, {"day": 0}, 0)
    assert res.metrics["trafo_overload_kvah"] > 0
    load = env._h["trafo"][: env.k]
    names = list(env.net.trafo.name)
    assert np.isfinite(load[:, names.index("TR_4T")]).all()
    assert np.isnan(np.delete(load, names.index("TR_4T"), axis=1)).all()
    k = evaluate_policies(env, {"u": ag}, days=[0], seed=0, keep_episodes=False).kpis
    assert k["trafo_overload_kvah"].iloc[0] > 0
    with pytest.raises(ValueError, match="unknown transformers"):
        _env(csv, data, _problem(trafo_limits_kva={"TR_X": 1.0}), [])


def test_agent_controlled_battery_follows_its_setpoint(csv, data):
    p = _problem(
        battery_policy="agent",
        scenario=dict(n_chargers=5, added_pv_kwp=2000.0, battery_kwh=1450.0),
    )
    env = _env(csv, data, p, [])
    assert env.action_space.shape == (6,)
    soc = {}
    for name, a in (("charge", 1.0), ("idle", 0.5), ("discharge", 0.0)):
        env.reset(seed=0, options={"day": 0})
        s0 = env.e_kwh
        for _ in range(3):
            env.step(np.append(np.zeros(5), a))
        soc[name] = env.e_kwh - s0
    assert soc["charge"] > 0 > soc["discharge"] and abs(soc["idle"]) < 1e-6
    # a policy that gives only the charger actions leaves the battery to the rule
    env.reset(seed=0, options={"day": 0})
    env.step(np.zeros(5))
    assert 0.0 <= env._bat_action <= 1.0


def test_oracle_dispatches_the_battery_when_the_agent_controls_it(csv, data):
    p = _problem(
        battery_policy="agent",
        scenario=dict(n_chargers=5, added_pv_kwp=2000.0, battery_kwh=1450.0),
    )
    env = _env(csv, data, p, _events())
    oracle = OracleAgent(p)
    oracle.bind(env)
    obs, _ = env.reset(seed=0, options={"day": 0})
    a = oracle.predict(obs)
    assert a.shape == (6,) and ((0 <= a) & (a <= 1)).all()


def test_grid_guard_gives_all_actions_and_discharges_above_the_limit(csv, data):
    p = _problem(
        battery_policy="agent",
        scenario=dict(n_chargers=5, added_pv_kwp=2000.0, battery_kwh=1450.0),
        trafo_limits_kva={"TR_4T": 1.0},
    )
    env = _env(csv, data, p, [])
    guard = GridGuardAgent(p)
    guard.bind(env)
    obs, _ = env.reset(seed=0, options={"day": 0})
    for _ in range(12):  # midday: the 4T flow is far above 1 kVA, import or export
        a = guard.predict(obs)
        obs, *_ = env.step(a)
    assert a.shape == (6,)
    assert abs(a[-1] - 0.5) > 0.05  # the battery acts against the transformer flow
    # without a limit it follows the self-consumption rule and still returns 6 entries
    p2 = _problem(
        battery_policy="agent",
        scenario=dict(n_chargers=5, added_pv_kwp=2000.0, battery_kwh=1450.0),
    )
    env2 = _env(csv, data, p2, [])
    g2 = GridGuardAgent(p2)
    g2.bind(env2)
    obs, _ = env2.reset(seed=0, options={"day": 0})
    assert g2.predict(obs).shape == (6,)


def test_site_rank_and_count_observations(csv, data):
    p = _problem()
    events = [_ev("ev_c04", 8, 1, 22.0), _ev("ev_c05", 8, 4, 22.0)]
    env = _env(csv, data, p, events)
    env.reset(seed=0, options={"day": 0})
    for _ in range(8):
        env.step(np.zeros(5))
    s = env._observe_state()
    rank = OBSERVATION_REGISTRY["charger_site_rank"][1](s, {})
    count = OBSERVATION_REGISTRY["charger_site_count"][1](s, {})
    assert rank[3] == 0.0 and rank[4] == 1.0  # the car with less slack goes first
    assert rank[0] == 1.0 and count[0] == 0.0  # empty charger
    assert count[3] == count[4] == 1.0  # both chargers of bus b4 are occupied


def test_warm_start_clones_the_rule(csv, data):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    from app.case_studies.kfki_campus.rl.train import Setup, collect_demos
    from app.operation.rl.agent import BasicPPOAgent

    p = _problem(agent=AgentSpec(n_steps=64, batch_size=32, n_envs=1, log_std_init=-1.0))
    setup = Setup(csv, data, START, END, _events())
    rule = PVFollowingAgent(p)
    p.events = EventSpec(source="file", session_file="x")
    obs, act, ret = collect_demos(p, setup, rule, n_days=2)
    assert obs.shape[0] == act.shape[0] == ret.shape[0] and act.shape[1] == 5
    agent = BasicPPOAgent(p)
    agent.prepare(DummyVecEnv([lambda: KfkiChargingEnv(p, csv, data, START, END, _events(), "train")]))

    def mse():
        pred = np.array([agent.predict(o, deterministic=True) for o in obs])
        return float(np.mean((pred - act) ** 2))

    before = mse()
    info = agent.pretrain(obs, act, ret, epochs=60)
    assert mse() < before and info["samples"] == len(obs)
