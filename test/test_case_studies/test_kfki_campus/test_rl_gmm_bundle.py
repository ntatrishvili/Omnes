"""Charging events from ``.gmm`` bundles (mixture + scaler), see ``rl/gmm_events.py``."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
import joblib  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from app.case_studies.kfki_campus.rl.gmm_events import (  # noqa: E402
    FEATURES,
    GmmBundleGenerator,
    WORKDAY_FILE,
    BundleSampler,
    load_bundle,
)
from app.operation.rl.events import GENERATOR_REGISTRY  # noqa: E402

CHARGERS = {f"c{i}": 22.0 for i in range(4)}


def _bundle(path, start_h=9.0, shift_h=4, seed=0, features=FEATURES):
    rng = np.random.default_rng(seed)
    n = 400
    start = rng.normal(start_h, 1.5, n)
    energy = np.clip(rng.lognormal(np.log(12.0), 0.5, n), 2.0, 60.0)
    stay = energy / 7.0 + rng.gamma(2.0, 1.0, n)
    raw = np.column_stack([((start - shift_h) % 24) * 3600.0, energy, stay])
    scaler = StandardScaler().fit(raw)
    gmm = GaussianMixture(2, random_state=seed).fit(scaler.transform(raw))
    joblib.dump(
        dict(gmm=gmm, scaler=scaler, features=list(features), time_shift_hour=shift_h),
        path,
    )
    return path


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("gmm")
    return _bundle(d / "wd.gmm", 9.0), _bundle(d / "we.gmm", 12.0, seed=1)


def _gen(files, **kw):
    return GmmBundleGenerator(str(files[0]), str(files[1]), **kw)


def test_sampler_undoes_the_scaler_and_the_time_shift(files):
    s = BundleSampler(load_bundle(files[0]))
    hour, energy, stay = s.sample(2000, np.random.default_rng(0))
    assert abs(np.median(hour) - 9.0) < 0.5  # shifted by 4 h in the file, back here
    assert ((0 <= hour) & (hour < 24)).all()
    assert (energy >= 1.0).all() and (energy <= 80.0).all() and (stay >= 0.25).all()


def test_generator_is_registered_and_reproducible(files):
    assert GENERATOR_REGISTRY["gmm_bundle"] is GmmBundleGenerator
    g = _gen(files)
    a = g.sample(np.random.default_rng(3), pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-04"), CHARGERS)
    b = g.sample(np.random.default_rng(3), pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-04"), CHARGERS)
    assert a and [e.event_id for e in a] == [e.event_id for e in b]
    assert [e.energy_kwh for e in a] == [e.energy_kwh for e in b]


def test_one_vehicle_per_charger_and_window_filter(files):
    g = _gen(files, workday_rate=8.0)
    start, end = pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-03")
    ev = g.sample(np.random.default_rng(1), start, end, CHARGERS)
    for cid in CHARGERS:
        mine = sorted((e for e in ev if e.charger_id == cid), key=lambda e: e.plug_in)
        assert all(a.plug_out <= b.plug_in for a, b in zip(mine, mine[1:]))
    assert all(e.plug_out > start and e.plug_in < end for e in ev)


def test_weekend_uses_the_weekend_rate(files):
    g = _gen(files, workday_rate=6.0, weekend_rate=0.0, holidays=False)
    sat = g.sample(np.random.default_rng(0), pd.Timestamp("2025-06-07"), pd.Timestamp("2025-06-08"), CHARGERS)
    mon = g.sample(np.random.default_rng(0), pd.Timestamp("2025-06-09"), pd.Timestamp("2025-06-10"), CHARGERS)
    assert len(sat) == 0 and len(mon) > 0  # (the day before a Saturday is a Friday: no spill)


def test_local_clock_is_one_hour_earlier_in_summer_and_equal_in_winter(files):
    for day, shift in (("2025-06-02", -1.0), ("2025-01-13", 0.0)):
        t0, t1 = pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1)
        model = _gen(files, clock="model").sample(np.random.default_rng(5), t0, t1, CHARGERS)
        local = _gen(files, clock="local").sample(np.random.default_rng(5), t0, t1, CHARGERS)
        by_id = {e.event_id: e for e in model}
        pairs = [(by_id[e.event_id], e) for e in local if e.event_id in by_id]
        assert pairs
        for m, loc in pairs:
            assert (loc.plug_in - m.plug_in) / pd.Timedelta("1h") == pytest.approx(shift)


def test_stay_and_energy_scales(files):
    t0, t1 = pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-03")
    base = _gen(files).sample(np.random.default_rng(2), t0, t1, {"c0": 22.0})
    scaled = _gen(files, stay_scale=0.5, energy_scale=2.0).sample(np.random.default_rng(2), t0, t1, {"c0": 22.0})
    first = {e.event_id: e for e in base}
    for e in scaled:
        if e.event_id in first:
            assert e.connected_h == pytest.approx(0.5 * first[e.event_id].connected_h)
            assert e.energy_kwh == pytest.approx(2.0 * first[e.event_id].energy_kwh)


def test_bad_bundles_are_rejected(tmp_path):
    joblib.dump({"gmm": 1}, tmp_path / "x.gmm")
    with pytest.raises(ValueError, match="not a gmm bundle"):
        load_bundle(tmp_path / "x.gmm")
    bad = _bundle(tmp_path / "y.gmm", features=["a", "b", "c"])
    with pytest.raises(ValueError, match="features"):
        load_bundle(bad)
    with pytest.raises(ValueError, match="clock"):
        GmmBundleGenerator(clock="utc")


@pytest.mark.skipif(not WORKDAY_FILE.exists(), reason="campus .gmm files not present")
def test_the_campus_mixtures_give_plausible_sessions():
    g = GmmBundleGenerator()
    ev = g.sample(
        np.random.default_rng(0), pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-03"), {f"c{i}": 22.0 for i in range(9)}
    )
    ev = [e for e in ev if e.plug_in >= pd.Timestamp("2025-06-02")]
    assert 5 <= len(ev) <= 25
    assert all(1.0 <= e.energy_kwh <= 80.0 and e.connected_h >= 0.25 for e in ev)
    assert 5 <= np.median([e.plug_in.hour for e in ev]) <= 14  # a working day
