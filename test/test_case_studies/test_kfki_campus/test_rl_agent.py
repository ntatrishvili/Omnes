"""RL agents on the KFKI environment (Stable-Baselines3 tests are skipped if it is missing)."""

import numpy as np
import pytest

from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.rl.env import KfkiChargingEnv
from app.case_studies.kfki_campus.rl.evaluate import evaluate_policies
from app.case_studies.kfki_campus.synthetic import make_synthetic_csv
from app.operation.rl.agent import RandomAgent, make_agent
from app.operation.rl.problem import AgentSpec, EpisodeSpec, RLProblem


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    data = load_campus_data()
    csv = make_synthetic_csv(
        tmp_path_factory.mktemp("agent") / "ts.csv",
        start="2025-06-02",
        days=3,
        data=data,
    )
    problem = RLProblem(
        scenario=dict(n_chargers=2),
        power_flow_mode="post",
        episode=EpisodeSpec(train_fraction=0.67),
        agent=AgentSpec(total_timesteps=128, n_steps=64, batch_size=32, n_envs=1),
    )
    return KfkiChargingEnv(problem, csv, data, "2025-06-02", "2025-06-05")


def test_random_agent_actions_are_in_bounds(env):
    agent = RandomAgent(env.problem, seed=1)
    agent.bind(env)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        a = agent.predict(obs)
        assert a.shape == env.action_space.shape and ((0 <= a) & (a <= 1)).all()


def test_unknown_algorithm_is_rejected(env):
    problem = RLProblem(agent=AgentSpec(algo="NOPE"))
    with pytest.raises(ValueError):
        make_agent(problem)


def test_ppo_agent_trains_saves_and_acts(env, tmp_path):
    pytest.importorskip("stable_baselines3")
    from app.operation.rl.agent import BasicPPOAgent

    agent = make_agent(env.problem)
    assert isinstance(agent, BasicPPOAgent)
    agent.train(env, 128)
    obs, _ = env.reset(seed=0)
    a = agent.predict(obs)
    assert env.action_space.contains(np.asarray(a, dtype=np.float32))
    agent.save(tmp_path / "ppo")
    loaded = BasicPPOAgent.load(tmp_path / "ppo.zip", env.problem)
    assert np.allclose(loaded.predict(obs), a)
    ev = evaluate_policies(env, {"ppo": loaded}, days=[0], seed=0)
    assert len(ev.kpis) == 1 and np.isfinite(ev.kpis["reward"]).all()
