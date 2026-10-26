"""KFKI RL environment: feasibility, alignment with the optimisation pipeline, sessions, baselines."""

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.rl.baselines import (
    OracleAgent,
    PVFollowingAgent,
    UncontrolledAgent,
)
from app.case_studies.kfki_campus.rl.env import KfkiChargingEnv
from app.case_studies.kfki_campus.rl.evaluate import evaluate_policies
from app.case_studies.kfki_campus.synthetic import make_synthetic_csv
from app.operation.rl.events import ChargingEvent
from app.operation.rl.problem import EpisodeSpec, EventSpec, RewardTerm, RLProblem
from app.operation.rl.rollout import run_episode

START, END = "2025-06-02 00:00", "2025-06-05 00:00"


@pytest.fixture(scope="module")
def data():
    return load_campus_data()


@pytest.fixture(scope="module")
def csv(tmp_path_factory, data):
    return make_synthetic_csv(
        tmp_path_factory.mktemp("rl") / "ts.csv", start="2025-06-02", days=3, data=data
    )


def _problem(**kw):
    base = dict(
        scenario=dict(n_chargers=2, added_pv_kwp=2000.0, battery_kwh=1450.0),
        power_flow_mode="post",
        episode=EpisodeSpec(train_fraction=0.67),
        events=EventSpec(
            source="generator", generator_params=dict(arrivals_per_day=3.0)
        ),
    )
    base.update(kw)
    return RLProblem(**base)


def _env(csv, data, problem=None, events=None, split="train"):
    return KfkiChargingEnv(problem or _problem(), csv, data, START, END, events, split)


def _hour_events(chargers):
    """Aligned sessions: day 0, 08:00-12:00, 44 kWh on a 22 kW charger (2 h at full power)."""
    t = pd.Timestamp(START) + pd.Timedelta(hours=8)
    return [ChargingEvent(f"s_{c}", c, t, 4.0, 44.0) for c in chargers]


# ------------------------------------------------------------------ environment contract
def test_default_scenario_has_2009_kwp_of_pv(csv, data):
    env = _env(csv, data)
    assert env.scenario.added_pv_kwp + 9.0 == 2009.0
    assert env.action_space.shape == (2,)


def test_gymnasium_env_checker_passes(csv, data):
    check_env(_env(csv, data), skip_render_check=True)


def test_reset_is_reproducible_for_a_seed(csv, data):
    env = _env(csv, data)
    o1, _ = env.reset(seed=5, options={"day": 1})
    n1 = env.n_events
    o2, _ = env.reset(seed=5, options={"day": 1})
    assert env.n_events == n1 and np.allclose(o1, o2)


def test_actions_are_projected_to_feasible_power(csv, data):
    ev = _hour_events(["ev_c01"])
    env = _env(
        csv, data, _problem(events=EventSpec(source="file", session_file="x")), ev
    )
    env.reset(seed=0, options={"day": 0})
    for k in range(24):
        env.step(np.ones(2))
    p = env.ev_power()
    assert p["ev_c02"].sum() == 0  # no vehicle, no power
    assert p["ev_c01"].max() <= 22.0 + 1e-9
    assert p["ev_c01"].sum() == pytest.approx(44.0)  # exactly the requested energy
    assert (p["ev_c01"].iloc[8:10] == 22.0).all() and p["ev_c01"].iloc[10:].sum() == 0
    s = env.sessions()
    assert s["delivered_kwh"].iloc[0] == pytest.approx(44.0)


def test_unserved_energy_is_reported_at_departure(csv, data):
    env = _env(
        csv,
        data,
        _problem(events=EventSpec(source="file", session_file="x")),
        _hour_events(["ev_c01"]),
    )
    env.reset(seed=0, options={"day": 0})
    unmet = 0.0
    for k in range(24):
        _, _, _, _, info = env.step(np.zeros(2))  # never charge
        unmet += info["unmet_kwh"]
    assert unmet == pytest.approx(44.0)
    assert env.sessions()["delivered_kwh"].iloc[0] == 0.0


def test_laxity_safety_layer_delivers_the_energy(csv, data):
    p = _problem(events=EventSpec(source="file", session_file="x"))
    p.action.safety_layer = "laxity"
    env = _env(csv, data, p, _hour_events(["ev_c01"]))
    env.reset(seed=0, options={"day": 0})
    for k in range(24):
        env.step(np.zeros(2))
    assert env.sessions()["delivered_kwh"].iloc[0] == pytest.approx(44.0)


# ------------------------------------------------------------------ alignment with the pipeline
def test_reward_sum_equals_import_plus_export_over_reference_energy(csv, data):
    p = _problem(
        reward=[RewardTerm("import_kwh", 1.0), RewardTerm("export_kwh", 1.0)],
        reward_baseline="none",
    )
    env = _env(csv, data, p)
    res = run_episode(env, UncontrolledAgent(p), {"day": 1}, seed=3)
    d = env.dispatch()
    expected = -(d["p_import"].sum() + d["p_export"].sum()) / env.reward_fn.e_ref
    assert res.reward == pytest.approx(expected, rel=1e-9)


def test_energy_balance_matches_the_optimiser_definition(csv, data):
    env = _env(csv, data)
    run_episode(env, PVFollowingAgent(), {"day": 0}, seed=1)
    d = env.dispatch()
    lhs = d["pv"] + d["p_import"] + d["p_discharge"]
    assert np.allclose(lhs, d["demand"] + d["p_charge"] + d["p_export"], atol=1e-6)
    assert d["soc_kwh"].between(0, 1450 + 1e-6).all()


def test_step_and_post_power_flow_give_the_same_network_results(csv, data):
    ev = _hour_events(["ev_c01", "ev_c02"])
    out = {}
    for mode in ("step", "post"):
        p = _problem(
            power_flow_mode=mode, events=EventSpec(source="file", session_file="x")
        )
        env = _env(csv, data, p, ev)
        run_episode(env, UncontrolledAgent(p), {"day": 0}, seed=0)
        out[mode] = env.sim_result()
    assert np.allclose(out["step"].vm_pu, out["post"].vm_pu, atol=1e-5)
    assert np.allclose(
        out["step"].trafo_loading_pct, out["post"].trafo_loading_pct, atol=1e-4
    )


def test_uncontrolled_agent_charges_at_full_power_from_plug_in(csv, data):
    ev = _hour_events(["ev_c01", "ev_c02"])  # 44 kWh at 22 kW from 08:00: two full hours
    p = _problem(events=EventSpec(source="file", session_file="x"))
    env = _env(csv, data, p, ev)
    run_episode(env, UncontrolledAgent(p), {"day": 0}, seed=0)
    expected = np.zeros((24, 2))
    expected[8:10, :] = 22.0
    assert np.allclose(env.ev_power().to_numpy(), expected)


def test_rollout_can_be_written_back_into_the_model(csv, data):
    p = _problem(episode=EpisodeSpec(mode="full_period"))
    env = _env(csv, data, p)
    run_episode(env, UncontrolledAgent(p), {"full_period": True}, seed=0)
    env.write_back("rl_run")
    from app.model.run_view import RunView

    run = RunView(env.model, "rl_run", env.model.time_set)
    assert np.allclose(run["ev_c01"].p_cons.value, env.ev_power()["ev_c01"])


# ------------------------------------------------------------------ baselines and evaluation
def test_oracle_is_not_worse_than_uncontrolled_in_ev_cost(csv, data):
    p = _problem(
        reward=[
            RewardTerm("import_kwh"),
            RewardTerm("export_kwh"),
            RewardTerm("unmet_kwh", 5.0),
        ],
        reward_baseline="none",
        battery_policy="none",
    )
    env = _env(csv, data, p, split="test")
    ev = evaluate_policies(
        env,
        {"uncontrolled": UncontrolledAgent(p), "oracle": OracleAgent(p)},
        days=[0, 1],
        seed=0,
    )
    k = ev.kpis.groupby("policy")[["reward", "ev_unmet_kwh"]].mean()
    assert k.loc["oracle", "ev_unmet_kwh"] == pytest.approx(0.0, abs=1e-6)
    assert k.loc["oracle", "reward"] >= k.loc["uncontrolled", "reward"] - 1e-9
    assert {"SC", "SS", "ev_pv_share", "viol_frac"} <= set(ev.kpis.columns)


def test_same_seed_gives_all_policies_the_same_events(csv, data):
    p = _problem()
    env = _env(csv, data, p, split="test")
    ev = evaluate_policies(
        env, {"a": UncontrolledAgent(p), "b": PVFollowingAgent(p)}, days=[0, 1], seed=4
    )
    n = ev.kpis.pivot(index="day", columns="policy", values="ev_requested_kwh")
    assert np.allclose(n["a"], n["b"])
