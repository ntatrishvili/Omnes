"""Examples of event distributions (rl/examples) and the guide that quotes them."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.case_studies.kfki_campus.rl.examples import (
    ex_bootstrap,
    ex_kde_joint,
    ex_scipy_marginals,
    ex_weekday_generator,
)
from app.case_studies.kfki_campus.rl.examples._guards import (
    SafeSampler,
    check_sampler,
    demo_sessions,
    plot_sampler,
)
from app.case_studies.kfki_campus.rl.examples.run_custom_events import main as run_main
from app.operation.rl.events import GENERATOR_REGISTRY, ChargingEventGenerator

START = pd.Timestamp("2025-06-02")  # a Monday
END = START + pd.Timedelta(days=7)
CHARGERS = {"a": 22.0, "b": 22.0, "c": 11.0}
README = (
    Path(__file__).resolve().parents[3] / "app/case_studies/kfki_campus/rl/README_RL.md"
)
EXAMPLES = {
    "marginals": ex_scipy_marginals,
    "kde": ex_kde_joint,
    "bootstrap": ex_bootstrap,
    "weekday": ex_weekday_generator,
}


@pytest.fixture(params=sorted(EXAMPLES))
def generator(request):
    return EXAMPLES[request.param].make_generator()


def test_examples_return_event_generators(generator):
    assert isinstance(generator, ChargingEventGenerator)


def test_example_events_are_valid(generator):
    events = generator.sample(np.random.default_rng(0), START, END, CHARGERS)
    assert len(events) > 5
    for e in events:
        assert e.energy_kwh > 0 and e.connected_h > 0
        assert e.charger_id in CHARGERS
        assert e.plug_out > START and e.plug_in < END
    for cid in CHARGERS:
        ev = sorted((e for e in events if e.charger_id == cid), key=lambda e: e.plug_in)
        assert all(a.plug_out <= b.plug_in for a, b in zip(ev, ev[1:]))


def test_example_events_are_reproducible(generator):
    ids = lambda s: [  # noqa: E731
        (e.event_id, round(e.energy_kwh, 6))
        for e in generator.sample(np.random.default_rng(s), START, END, CHARGERS)
    ]
    assert ids(7) == ids(7)
    assert ids(7) != ids(8)


def test_registered_generator_name_can_be_used_in_a_problem():
    assert GENERATOR_REGISTRY["weekday"] is ex_weekday_generator.WeekdayGenerator


def test_weekday_generator_has_fewer_events_on_weekends():
    gen = ex_weekday_generator.make_generator()
    days = pd.Timestamp("2025-06-02")
    events = gen.sample(
        np.random.default_rng(1), days, days + pd.Timedelta(days=140), {"a": 22.0}
    )
    dow = pd.Series([e.plug_in.dayofweek for e in events])
    per_weekday = (dow < 5).sum() / 5
    per_weekend = (dow >= 5).sum() / 2
    assert per_weekend < 0.7 * per_weekday


def test_weekday_generator_uses_the_rate_of_each_charger():
    gen = ex_weekday_generator.make_generator(rates={"busy": 3.0, "quiet": 0.2})
    events = gen.sample(
        np.random.default_rng(2),
        START,
        START + pd.Timedelta(days=60),
        {"busy": 22.0, "quiet": 22.0},
    )
    n = pd.Series([e.charger_id for e in events]).value_counts()
    assert n["busy"] > 3 * n["quiet"]


# ------------------------------------------------------------------ the safety net
class _Bad:
    """Negative energies, hours beyond 24."""

    def sample(self, n, rng):
        energy = rng.normal(5.0, 10.0, n)  # about 30 % negative
        return rng.uniform(0, 30, n), energy, np.full(n, 2.0)


def test_safe_sampler_repairs_bad_samples():
    hour, energy, dwell = SafeSampler(_Bad()).sample(2000, np.random.default_rng(0))
    assert len(hour) == len(energy) == len(dwell) == 2000
    assert (energy > 0).all() and (dwell > 0).all()
    assert ((0 <= hour) & (hour < 24)).all()


def test_safe_sampler_rejects_wrong_length():
    class Short:
        def sample(self, n, rng):
            return np.zeros(n - 1), np.ones(n - 1), np.ones(n - 1)

    with pytest.raises(ValueError, match="length"):
        SafeSampler(Short()).sample(10, np.random.default_rng(0))


def test_safe_sampler_gives_up_on_hopeless_samplers():
    class Negative:
        def sample(self, n, rng):
            return np.zeros(n), -np.ones(n), np.ones(n)

    with pytest.raises(RuntimeError, match="invalid"):
        SafeSampler(Negative(), max_tries=3).sample(10, np.random.default_rng(0))


def test_check_and_plot_sampler(tmp_path):
    sampler = SafeSampler(ex_scipy_marginals.make_sampler())
    summary, corr = check_sampler(sampler, n=2000)
    assert list(summary.index) == ["start_hour", "energy_kwh", "connected_h"]
    assert summary["share_non_positive"].sum() == 0
    assert corr.shape == (3, 3)
    assert plot_sampler(sampler, tmp_path / "s.png", n=500).exists()


def test_kde_keeps_the_correlation_of_the_data():
    sessions = demo_sessions(800)
    gen = ex_kde_joint.KdeSampler(sessions)
    _, energy, dwell = SafeSampler(gen).sample(5000, np.random.default_rng(0))
    data_corr = np.corrcoef(
        np.log(sessions["energy_kwh"]), np.log(sessions["connected_h"])
    )[0, 1]
    assert data_corr > 0.2
    assert np.corrcoef(np.log(energy), np.log(dwell))[0, 1] == pytest.approx(
        data_corr, abs=0.1
    )


# ------------------------------------------------------------------ script and guide
def test_run_custom_events_script_writes_results(tmp_path):
    kpis = run_main(
        [
            "--generator",
            "marginals",
            "--days",
            "4",
            "--eval-days",
            "2",
            "--out",
            str(tmp_path),
        ]
    )
    assert {"uncontrolled", "pv_following", "random"} <= set(kpis["policy"])
    assert (tmp_path / "rl_kpis.csv").exists()


@pytest.mark.parametrize("module", sorted(EXAMPLES))
def test_guide_quotes_the_example_code_verbatim(module):
    """Code in README_RL.md between the markers is the code of the example file."""
    src = Path(EXAMPLES[module].__file__).read_text(encoding="utf-8")
    regions = re.findall(r"# \[start:(\w+)\]\n(.*?)# \[end:\1\]", src, flags=re.S)
    assert regions, f"{module} has no [start:..] markers"
    guide = README.read_text(encoding="utf-8")
    for name, code in regions:
        assert (
            code.strip() in guide
        ), f"{module}:{name} is not quoted verbatim in README_RL.md"
