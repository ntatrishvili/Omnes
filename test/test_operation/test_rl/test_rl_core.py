"""Generic RL building blocks: problem description, rewards, observations, events."""

import numpy as np
import pandas as pd
import pytest

from app.operation.rl.events import (
    ChargingEvent,
    DummyChargingEventGenerator,
    JointDistributionGenerator,
    SessionFileGenerator,
    drop_overlaps,
)
from app.operation.rl.observations import (
    OBSERVATION_REGISTRY,
    ObsState,
    build_observation,
    observation_size,
)
from app.operation.rl.problem import ObservationTerm, RewardTerm, RLProblem
from app.operation.rl.rewards import RewardFunction

START = pd.Timestamp("2025-06-02")
END = pd.Timestamp("2025-06-09")


# ------------------------------------------------------------------ problem
def test_default_problem_is_valid():
    p = RLProblem()
    assert p.reward_weights()["unmet_kwh"] > p.reward_weights()["import_kwh"]
    assert p.events.source == "generator"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(power_flow_mode="sometimes"),
        dict(reward=[RewardTerm("not_a_metric")]),
        dict(reward=[RewardTerm("import_kwh", -1.0)]),
        dict(reward=[]),
        dict(observation=[ObservationTerm("nope")]),
        dict(reward_scale=0.0),
    ],
)
def test_invalid_problem_raises(kwargs):
    with pytest.raises(ValueError):
        RLProblem(**kwargs)


def test_from_dict_builds_nested_problem():
    p = RLProblem.from_dict(
        {
            "id": "x",
            "episode": {"episode_days": 2},
            "events": {"source": "file", "session_file": "s.csv"},
            "reward": [{"metric": "import_kwh", "weight": 2.0}],
        }
    )
    assert p.episode.episode_days == 2 and p.events.source == "file"
    assert [t.metric for t in p.reward] == ["import_kwh"]


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        RLProblem.from_dict({"episode": {"episodes_days": 2}})
    with pytest.raises(ValueError):
        RLProblem.from_dict({"nonsense": 1})


# ------------------------------------------------------------------ rewards
def _ctx(**kw):
    ctx = dict(
        dt_h=1.0,
        import_kwh=0.0,
        export_kwh=0.0,
        unmet_kwh=0.0,
        p_ev=np.zeros(2),
        p_ev_prev=np.zeros(2),
        p_bat_ch=0.0,
        p_bat_dis=0.0,
        clipped_kw=0.0,
        trafo_sn_kva=np.array([1000.0]),
        line_sn_kva=np.array([100.0]),
    )
    ctx.update(kw)
    return ctx


def test_reward_is_weighted_negative_cost_over_reference_energy():
    fn = RewardFunction(
        [RewardTerm("import_kwh", 1.0), RewardTerm("unmet_kwh", 5.0)],
        e_ref=10.0,
        baseline="none",
    )
    r, parts = fn(_ctx(import_kwh=20.0, unmet_kwh=2.0))
    assert r == pytest.approx(-(20.0 + 10.0) / 10.0)
    assert parts["unmet_kwh"] == pytest.approx(-1.0)


def test_exogenous_baseline_is_policy_independent_offset():
    terms = [RewardTerm("import_kwh", 1.0), RewardTerm("export_kwh", 2.0)]
    with_base = RewardFunction(terms, 10.0, "exogenous")
    no_base = RewardFunction(terms, 10.0, "none")
    ctx = _ctx(import_kwh=5.0, export_kwh=1.0)
    r1, _ = with_base(ctx, ref=(8.0, 3.0))
    r0, _ = no_base(ctx, ref=(8.0, 3.0))
    assert r1 - r0 == pytest.approx((8.0 + 2.0 * 3.0) / 10.0)


def test_overload_and_voltage_terms_need_a_power_flow():
    fn = RewardFunction(
        [RewardTerm("voltage_excess_pu", 10.0), RewardTerm("trafo_overload_kvah", 1.0)],
        1.0,
        "none",
    )
    assert fn(_ctx())[0] == 0.0  # no power flow in the context
    ctx = _ctx(vm_pu=np.array([1.00, 1.07]), trafo_loading_pct=np.array([120.0]))
    r, parts = fn(ctx)
    assert parts["voltage_excess_pu"] == pytest.approx(-10.0 * 0.02)
    assert parts["trafo_overload_kvah"] == pytest.approx(-0.2 * 1000.0)


def test_hinge_transform():
    fn = RewardFunction(
        [RewardTerm("clipped_kw", 1.0, "hinge", threshold=2.0)], 1.0, "none"
    )
    assert fn(_ctx(clipped_kw=1.0))[0] == 0.0
    assert fn(_ctx(clipped_kw=5.0))[0] == pytest.approx(-3.0)


# ------------------------------------------------------------------ observations
def _state(n=2, t=5, mode="perfect"):
    idx = pd.date_range("2025-06-01", periods=48, freq="1h")
    return ObsState(
        t=t,
        index=idx,
        series={"pv_kw": np.arange(48.0), "base_load_kw": np.full(48, 100.0)},
        scale={"pv_kw": 48.0, "base_load_kw": 100.0},
        steps_per_day=24,
        dt_h=1.0,
        forecast_mode=mode,
        soc=0.5,
        trafo_loading_prev=40.0,
        v_max_prev=1.02,
        connected=np.array([True, False][:n]),
        remaining_kwh=np.array([10.0, 0.0][:n]),
        steps_left=np.array([4, 0][:n]),
        rated_kw=np.full(n, 22.0),
        p_prev=np.zeros(n),
    )


def test_every_term_has_the_announced_size():
    s = _state()
    for name, (size, fn) in OBSERVATION_REGISTRY.items():
        out = fn(s, {})
        assert out.shape == (size(s.n_chargers, {}),), name
        assert np.isfinite(out).all(), name


def test_observation_vector_concatenates_terms():
    terms = [
        ObservationTerm("pv_now"),
        ObservationTerm("pv_forecast", {"k": 4}),
        ObservationTerm("charger_connected"),
    ]
    obs = build_observation(terms, _state())
    assert obs.shape == (observation_size(terms, 2),) == (1 + 4 + 2,)
    assert obs.dtype == np.float32


def test_persistence_forecast_uses_previous_day_and_perfect_the_future():
    s = _state(t=30)
    perfect = OBSERVATION_REGISTRY["pv_forecast"][1](s, {"k": 2})
    assert np.allclose(perfect, np.array([31, 32]) / 48.0)
    s = _state(t=30, mode="persistence")
    pers = OBSERVATION_REGISTRY["pv_forecast"][1](s, {"k": 2})
    assert np.allclose(pers, np.array([7, 8]) / 48.0)


# ------------------------------------------------------------------ events
def _sample(seed, **kw):
    gen = DummyChargingEventGenerator(**kw)
    return gen.sample(np.random.default_rng(seed), START, END, {"a": 22.0, "b": 11.0})


def test_dummy_generator_is_reproducible_and_seed_dependent():
    a, b, c = _sample(1), _sample(1), _sample(2)
    assert [e.event_id for e in a] == [e.event_id for e in b]
    assert [e.energy_kwh for e in a] == [e.energy_kwh for e in b]
    assert [e.energy_kwh for e in a] != [e.energy_kwh for e in c]


def test_dummy_generator_events_are_valid_and_do_not_overlap():
    events = _sample(3, arrivals_per_day=4.0)
    assert events
    for cid in ("a", "b"):
        ev = sorted((e for e in events if e.charger_id == cid), key=lambda e: e.plug_in)
        assert all(x.plug_out <= y.plug_in for x, y in zip(ev, ev[1:]))
    assert all(e.energy_kwh > 0 and e.connected_h > 0 for e in events)
    assert all(e.plug_out > START and e.plug_in < END for e in events)


def test_drop_overlaps_keeps_the_earlier_session():
    e1 = ChargingEvent("1", "a", START, 4.0, 10.0)
    e2 = ChargingEvent("2", "a", START + pd.Timedelta(hours=2), 2.0, 5.0)
    e3 = ChargingEvent("3", "a", START + pd.Timedelta(hours=5), 1.0, 5.0)
    assert [e.event_id for e in drop_overlaps([e2, e1, e3])] == ["1", "3"]


def test_session_file_generator_filters_by_window_and_charger():
    inside = ChargingEvent("in", "a", START + pd.Timedelta(hours=1), 2.0, 5.0)
    carry = ChargingEvent("carry", "a", START - pd.Timedelta(hours=1), 3.0, 5.0)
    out = ChargingEvent("out", "a", END + pd.Timedelta(hours=1), 2.0, 5.0)
    other = ChargingEvent("other", "z", START + pd.Timedelta(hours=1), 2.0, 5.0)
    gen = SessionFileGenerator([inside, carry, out, other])
    got = gen.sample(None, START, END, {"a": 22.0})
    assert {e.event_id for e in got} == {"in", "carry"}


def test_joint_generator_needs_a_sampler_and_uses_it():
    with pytest.raises(NotImplementedError):
        JointDistributionGenerator().sample(
            np.random.default_rng(0), START, END, {"a": 22.0}
        )

    class Sampler:
        def sample(self, n, rng):
            return np.full(n, 8.0), np.full(n, 12.0), np.full(n, 3.0)

    gen = JointDistributionGenerator(Sampler(), arrivals_per_day=1.0)
    events = gen.sample(np.random.default_rng(0), START, END, {"a": 22.0})
    assert events and all(e.energy_kwh == 12.0 and e.connected_h == 3.0 for e in events)
