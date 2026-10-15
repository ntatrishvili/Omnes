"""The own-files recipe: ``.gmm`` mixtures for random events and a session file for fixed ones."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.mixture import GaussianMixture  # noqa: E402

from app.case_studies.kfki_campus.rl.examples import ex_gmm_files  # noqa: E402
from app.case_studies.kfki_campus.rl.examples.run_custom_events import (  # noqa: E402
    main as run_main,
)
from app.case_studies.kfki_campus.rl.problem_io import (  # noqa: E402
    load_problem,
    write_problem,
)
from app.case_studies.kfki_campus.rl.sessions import write_sessions  # noqa: E402
from app.operation.rl.events import (  # noqa: E402
    GENERATOR_REGISTRY,
    DummyChargingEventGenerator,
)
from app.operation.rl.problem import EventSpec, RLProblem  # noqa: E402

START = pd.Timestamp("2025-06-02")  # a Monday
README = (
    Path(__file__).resolve().parents[3] / "app/case_studies/kfki_campus/rl/README_RL.md"
)


@pytest.fixture(scope="module")
def gmm_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("gmm")
    return (
        ex_gmm_files.fit_demo_gmm(d / "workday.gmm", weekend=False, seed=1),
        ex_gmm_files.fit_demo_gmm(d / "weekend.gmm", weekend=True, seed=2),
    )


def test_load_gmm_returns_the_fitted_mixture(gmm_files):
    gmm = ex_gmm_files.load_gmm(gmm_files[0])
    assert isinstance(gmm, GaussianMixture) and gmm.means_.shape[1] == 3


def test_load_gmm_rejects_wrong_dimension_and_wrong_objects(tmp_path):
    import joblib

    two_d = GaussianMixture(2, random_state=0).fit(
        np.random.default_rng(0).normal(size=(50, 2))
    )
    joblib.dump(two_d, tmp_path / "two.gmm")
    with pytest.raises(ValueError, match="3 dimensions"):
        ex_gmm_files.load_gmm(tmp_path / "two.gmm")
    joblib.dump({"not": "a gmm"}, tmp_path / "dict.gmm")
    with pytest.raises(ValueError, match="GaussianMixture"):
        ex_gmm_files.load_gmm(tmp_path / "dict.gmm")


def test_sampler_follows_our_rng_and_does_not_repeat(gmm_files):
    sampler = ex_gmm_files.GmmSampler(ex_gmm_files.load_gmm(gmm_files[0]))
    a = sampler.sample(5, np.random.default_rng(1))
    b = sampler.sample(5, np.random.default_rng(1))
    assert all(
        np.allclose(x, y) for x, y in zip(a, b)
    )  # same rng state -> same numbers
    rng = np.random.default_rng(1)
    c, d = sampler.sample(5, rng), sampler.sample(5, rng)
    assert not np.allclose(c[1], d[1])  # successive calls differ (not a frozen seed)


def test_generator_events_are_valid_although_the_mixture_is_raw(gmm_files):
    gen = ex_gmm_files.make_generator(*gmm_files, workday_rate=2.0, weekend_rate=0.5)
    end = START + pd.Timedelta(days=28)
    events = gen.sample(np.random.default_rng(0), START, end, {"a": 22.0, "b": 22.0})
    assert len(events) > 40
    assert all(e.energy_kwh > 0 and e.connected_h > 0 for e in events)
    hours = [e.plug_in.hour + e.plug_in.minute / 60 for e in events]
    assert min(hours) >= 0 and max(hours) < 24
    dow = pd.Series([e.plug_in.dayofweek for e in events])
    assert (dow >= 5).sum() / 2 < 0.6 * (dow < 5).sum() / 5  # fewer events on weekends


def test_generator_is_reproducible(gmm_files):
    gen = ex_gmm_files.make_generator(*gmm_files)
    ids = lambda s: [  # noqa: E731
        (e.event_id, round(e.energy_kwh, 6))
        for e in gen.sample(
            np.random.default_rng(s), START, START + pd.Timedelta(days=7), {"a": 22.0}
        )
    ]
    assert ids(3) == ids(3) and ids(3) != ids(4)


def test_arrival_rates_from_a_session_table():
    rows = []
    for d in range(14):  # two full weeks from a Monday
        day = START + pd.Timedelta(days=d)
        n = 1 if day.dayofweek >= 5 else 3  # per charger
        for cid in ("a", "b"):
            rows += [
                {"charger_id": cid, "plug_in": day + pd.Timedelta(hours=8 + k)}
                for k in range(n)
            ]
    workday, weekend = ex_gmm_files.arrival_rates(pd.DataFrame(rows))
    assert workday == pytest.approx(3.0) and weekend == pytest.approx(1.0)


def test_generator_can_be_named_in_the_workbook(gmm_files, tmp_path):
    """Paths in ``generator_params`` survive the workbook, so the problem can be file-driven."""
    params = {
        "workday": str(gmm_files[0]),
        "weekend": str(gmm_files[1]),
        "workday_rate": 2.0,
        "weekend_rate": 0.5,
    }
    problem = RLProblem(
        events=EventSpec(
            source="generator", generator="gmm_files", generator_params=params
        )
    )
    loaded = load_problem(write_problem(problem, tmp_path / "rl.xlsx"))
    assert (
        loaded.events.generator == "gmm_files"
        and loaded.events.generator_params == params
    )
    gen = GENERATOR_REGISTRY[loaded.events.generator](**loaded.events.generator_params)
    assert gen.sample(
        np.random.default_rng(0), START, START + pd.Timedelta(days=3), {"a": 22.0}
    )


def test_script_trains_on_gmm_events_and_tests_on_the_session_file(gmm_files, tmp_path):
    """The recipe end to end: GMM files for training, fixed sessions for the evaluation."""
    chargers = {f"ev_c0{i}": 22.0 for i in (1, 2, 3)}
    sessions = DummyChargingEventGenerator().sample(
        np.random.default_rng(5), START, START + pd.Timedelta(days=4), chargers
    )
    session_file = write_sessions(sessions, tmp_path / "sessions.csv")
    kpis = run_main(
        [
            "--generator",
            "gmm",
            "--gmm-workday",
            str(gmm_files[0]),
            "--gmm-weekend",
            str(gmm_files[1]),
            "--session-file",
            str(session_file),
            "--events",
            "mixed",
            "--eval-events",
            "file",
            "--days",
            "4",
            "--eval-days",
            "2",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert {"uncontrolled", "pv_following"} <= set(kpis["policy"])
    # evaluation on the fixed file: both policies see exactly the same sessions
    n = kpis.pivot(index="day", columns="policy", values="ev_requested_kwh")
    assert np.allclose(n["uncontrolled"], n["pv_following"])
    assert (tmp_path / "out" / "rl_kpis.csv").exists()


def test_script_asks_for_the_gmm_files_and_the_session_file(tmp_path):
    with pytest.raises(SystemExit):
        run_main(["--generator", "gmm", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        run_main(
            ["--generator", "marginals", "--events", "file", "--out", str(tmp_path)]
        )


def test_guide_quotes_the_gmm_example_verbatim():
    src = Path(ex_gmm_files.__file__).read_text(encoding="utf-8")
    regions = re.findall(r"# \[start:(\w+)\]\n(.*?)# \[end:\1\]", src, flags=re.S)
    assert {n for n, _ in regions} == {"load", "sampler", "rates", "generator"}
    guide = README.read_text(encoding="utf-8")
    for name, code in regions:
        assert (
            code.strip() in guide
        ), f"ex_gmm_files:{name} is not quoted verbatim in README_RL.md"
