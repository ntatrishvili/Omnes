import numpy as np
import pandas as pd
import pytest
from pandas import date_range

from app.model.timeseries_object import TimeseriesObject


# nosec B101
@pytest.fixture
def sample_df():
    index = date_range("2022-01-01", periods=96, freq="15min")
    data = np.random.normal(10, 1, size=96)
    return pd.DataFrame(data, index=index, columns=["load"])


def test_init_from_dataframe(sample_df):
    ts = TimeseriesObject(data=sample_df)
    assert not ts.data.empty
    assert isinstance(ts.data, pd.DataFrame)
    assert ts.freq == "15min"
    assert len(ts.data) == 96


def test_resample_to_1h(sample_df):
    ts = TimeseriesObject(data=sample_df)
    resampled = ts.to_1h(closed="left")
    assert isinstance(resampled, TimeseriesObject)
    assert resampled.freq == "1h"
    assert len(resampled.data) == 24

    resampled = ts.to_1h(closed="right")
    assert resampled.freq == "1h"
    assert len(resampled.data) == 25


def test_resample_to_upsample(sample_df):
    ts = TimeseriesObject(data=sample_df)
    down = ts.to_1h(closed="right")
    up = down.to_15m(closed="left")
    assert isinstance(up, TimeseriesObject)
    assert up.data.shape[0] == 97
    down = ts.to_1h(closed="left")
    up = down.to_15m(closed="left")
    assert up.data.shape[0] == 93


def test_getattr_proxy(sample_df):
    ts = TimeseriesObject(data=sample_df)
    assert ts.sum().iloc[0] > 0  # Should proxy to DataFrame.sum()
    assert ts.mean().iloc[0] > 0  # Should proxy to DataFrame.mean()
    assert ts.min().iloc[0] >= 0  # Should proxy to DataFrame.min()


def test_to_nd(sample_df):
    ts = TimeseriesObject(data=sample_df)
    arr = ts.to_nd()
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 96


def test_empty_to_pulp(monkeypatch):
    from app.model import timeseries_object

    monkeypatch.setattr(
        timeseries_object,
        "create_empty_pulp_var",
        lambda name, time_set: [f"{name}_{i}" for i in range(time_set)],
    )

    ts = TimeseriesObject()
    result = ts.to_pulp("foo", "1H", 10)
    assert result == [f"foo_{i}" for i in range(10)]


def test_partial_to_pulp(sample_df, monkeypatch):
    df = sample_df.iloc[:48]
    ts = TimeseriesObject(data=df)
    monkeypatch.setattr(
        "app.model.timeseries_object.create_empty_pulp_var",
        lambda name, time_set: [f"{name}_{i}" for i in range(time_set)],
    )

    result = ts.to_pulp("foo", "15min", 96)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 48
