import pytest
import numpy as np
import pandas as pd
import xarray as xr
from app.infra.timeseries_object import TimeseriesObject


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {"val": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="h")
    )


@pytest.fixture
def multi_col_df():
    return pd.DataFrame(
        {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]},
        index=pd.date_range("2020-01-01", periods=3, freq="1h"),
    )


@pytest.fixture
def multi_data():
    times = pd.date_range("2025-01-01", periods=24, freq="1h")
    rng = np.random.default_rng(42)
    return xr.DataArray(
        data=rng.random((24, 3, 2)),
        dims=["timestamp", "location", "scenario"],
        coords={
            "timestamp": times,
            "location": ["site_A", "site_B", "site_C"],
            "scenario": ["optimistic", "pessimistic"],
        },
        attrs={"technology_type": "pv", "entity_id": "pv_farm_1"},
    )


@pytest.fixture
def ts_multi(multi_data):
    return TimeseriesObject(data=multi_data)


@pytest.fixture
def simple_df2():
    return pd.DataFrame(
        {"power": [10, 15, 20, 25]},
        index=pd.date_range("2025-01-01", periods=4, freq="1h"),
    )


@pytest.fixture
def ts_simple(simple_df2):
    return TimeseriesObject(data=simple_df2)


def test_init_with_metadata():
    df = pd.DataFrame(
        {"power": [5, 10, 15]}, index=pd.date_range("2020-01-01", periods=3, freq="1h")
    )
    ts = TimeseriesObject(
        data=df, entity_id="pv1", technology_type="solar", location="building_A"
    )
    assert ts.get_metadata("entity_id") == "pv1"
    assert ts.get_metadata("technology_type") == "solar"
    assert ts.get_metadata("location") == "building_A"


def test_normalize_freq():
    assert TimeseriesObject.normalize_freq("h") == "1h"
    assert TimeseriesObject.normalize_freq("15min") == "15min"


def test_to_df_backward_compatibility(simple_df):
    ts = TimeseriesObject(data=simple_df)
    result_df = ts.to_df()
    assert isinstance(result_df, pd.DataFrame)
    assert np.array_equal(result_df.values.flatten(), [1, 2, 3])


def test_selection_operations(ts_multi):
    site_a_ts = ts_multi.sel(location="site_A")
    assert "location" not in site_a_ts.data.dims
    assert site_a_ts.data.sizes["scenario"] == 2
    subset_ts = ts_multi.sel(timestamp=slice("2025-01-01 06:00", "2025-01-01 12:00"))
    assert subset_ts.data.sizes["timestamp"] == 7


def test_integer_selection(ts_multi):
    first_timestep = ts_multi.isel(timestamp=0)
    assert "timestamp" not in first_timestep.data.dims
    first_three_times = ts_multi.isel(timestamp=slice(0, 3))
    assert first_three_times.data.sizes["timestamp"] == 3


def test_add_dimension(ts_simple):
    expanded_ts = ts_simple.add_dimension("weather", ["sunny", "cloudy"])
    assert "weather" in expanded_ts.data.dims
    assert expanded_ts.data.sizes["weather"] == 2


def test_metadata_operations(ts_simple):
    ts_with_meta = ts_simple.set_metadata(
        peak_power=100, efficiency=0.95, constraints={"max_power": "< 80kW"}
    )
    assert ts_with_meta.get_metadata("peak_power") == 100
    assert ts_with_meta.get_metadata("efficiency") == 0.95
    all_meta = ts_with_meta.get_metadata()
    assert "peak_power" in all_meta
    assert "constraints" in all_meta


def test_multi_dimensional_backward_compatibility(ts_multi):
    flat_array = ts_multi.to_nd()
    expected_size = 24 * 3 * 2
    assert len(flat_array) == expected_size
    values = ts_multi.value()
    assert len(values) == expected_size


def test_complex_energy_scenario():
    times = pd.date_range("2025-01-01", periods=8, freq="1h")
    rng = np.random.default_rng(123)
    pv_data = xr.DataArray(
        data=rng.random((8, 2, 3)),
        dims=["timestamp", "weather", "degradation_year"],
        coords={
            "timestamp": times,
            "weather": ["sunny", "cloudy"],
            "degradation_year": [0, 1, 2],
        },
        attrs={
            "technology_type": "pv",
            "peak_power_kw": 100,
            "location": "rooftop_commercial",
        },
    )
    ts_pv = TimeseriesObject(data=pv_data)
    avg_weather = ts_pv.data.mean(dim="weather")
    assert avg_weather.dims == ("timestamp", "degradation_year")
    year_0 = ts_pv.sel(degradation_year=0)
    assert "degradation_year" not in year_0.data.dims
    df_export = ts_pv.to_df()
    assert isinstance(df_export, pd.DataFrame)
    assert len(df_export) == 8
    assert df_export.shape[1] > 1


def test_empty_timeseriesobject():
    ts = TimeseriesObject()
    assert ts.empty()
    assert ts.freq is None
    assert isinstance(repr(ts), str)


def test_eq_and_neq():
    df = pd.DataFrame(
        {"a": [1, 2]}, index=pd.date_range("2020-01-01", periods=2, freq="h")
    )
    ts1 = TimeseriesObject(data=df)
    ts2 = TimeseriesObject(data=df)
    assert ts1 == ts2
    ts3 = TimeseriesObject()
    assert ts1 != ts3
    assert ts1 != 123


def test_getattr_delegation():
    df = pd.DataFrame(
        {"a": [1, 2]}, index=pd.date_range("2020-01-01", periods=2, freq="h")
    )
    ts = TimeseriesObject(data=df)
    assert hasattr(ts, "mean")
    with pytest.raises(AttributeError):
        _ = ts.nonexistent_attr


def test_to_1h_and_to_15m(ts_simple):
    # Should work
    ts_1h = ts_simple.to_1h()
    assert isinstance(ts_1h, TimeseriesObject)
    ts_15m = ts_simple.to_15m()
    assert isinstance(ts_15m, TimeseriesObject)
    # Should raise if freq is None
    ts_empty = TimeseriesObject()
    with pytest.raises(ValueError):
        ts_empty.to_1h()
    with pytest.raises(ValueError):
        ts_empty.to_15m()


def test_resample_to_in_place_and_errors(ts_simple):
    # in_place
    ts = TimeseriesObject(data=ts_simple.data)
    ts2 = ts.resample_to("1h", in_place=True)
    assert ts2 is ts
    # ffill and bfill methods
    ts_ffill = ts.resample_to("1h", method="agg")
    assert isinstance(ts_ffill, TimeseriesObject)
    ts_bfill = ts.resample_to("1h", method="bfill")
    assert isinstance(ts_bfill, TimeseriesObject)
    # Unsupported agg


def test_infer_freq_from_two_dates_and_short_data():
    arr = xr.DataArray(
        data=[1, 2],
        dims=["timestamp"],
        coords={"timestamp": pd.date_range("2020-01-01", periods=2, freq="15min")},
    )
    freq = TimeseriesObject._infer_frequency_from_data(TimeseriesObject(data=arr))
    assert freq == "15min"


def test_infer_freq_from_two_dates_error():
    # Should raise if no timestamp coord
    arr = xr.DataArray([1, 2], dims=["not_timestamp"])
    with pytest.raises(ValueError):
        from app.infra.timeseries_object import _infer_freq_from_two_dates

        _infer_freq_from_two_dates(arr)


def test_initialize_data_array_branches():
    # input_path and col but file does not exist
    with pytest.raises(FileNotFoundError):
        TimeseriesObject(input_path="notfound.csv", col="val")
    # No data, no input_path, no col
    ts = TimeseriesObject()
    assert isinstance(ts.data, xr.DataArray)


def test_initialize_frequency_with_freq_param():
    df = pd.DataFrame(
        {"a": [1, 2]}, index=pd.date_range("2020-01-01", periods=2, freq="1h")
    )
    ts = TimeseriesObject(data=df, freq="1h")
    print(ts.freq)
    assert ts.freq == "1h"


def test_get_values_branches(ts_simple):
    # freq != self.freq, time_set != resampled size
    arr = xr.DataArray(
        [1, 2, 3, 4],
        dims=["timestamp"],
        coords={"timestamp": pd.date_range("2020-01-01", periods=4, freq="h")},
    )
    ts = TimeseriesObject(data=arr)
    vals = ts.value(freq="15min", time_set=2)
    assert isinstance(vals, np.ndarray)
    # time_set != self.data.sizes["timestamp"]
    vals2 = ts.value(time_set=2)
    assert isinstance(vals2, np.ndarray)


def test_empty_dataframe_to_xarray():
    df = pd.DataFrame({"val": []}, index=pd.to_datetime([]))
    ts = TimeseriesObject(data=df)
    assert ts.empty()


def test_to_df_multidim():
    arr = xr.DataArray(
        data=np.ones((2, 2, 2)),
        dims=["timestamp", "a", "b"],
        coords={
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="h"),
            "a": [1, 2],
            "b": [3, 4],
        },
    )
    ts = TimeseriesObject(data=arr)
    df = ts.to_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2


def test_read_csv_to_dataframe_and_read(tmp_path):
    # Create a valid CSV
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame(
        {"timestamp": ["2020.01.01 00:00", "2020.01.01 01:00"], "val": [1, 2]}
    )
    df.to_csv(csv_path, sep=";", index=False)
    out_df = TimeseriesObject._read_csv_to_dataframe(str(csv_path), "val")
    assert list(out_df["val"]) == [1, 2]
    # File not found
    with pytest.raises(FileNotFoundError):
        TimeseriesObject._read_csv_to_dataframe("notfound.csv", "val")
    # Empty file
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("")
    with pytest.raises(ValueError):
        TimeseriesObject._read_csv_to_dataframe(str(empty_path), "val")
    # Missing column
    with pytest.raises(KeyError):
        TimeseriesObject._read_csv_to_dataframe(str(csv_path), "notacol")
    # Test static read method
    ts = TimeseriesObject.read(str(csv_path), "val")
    assert isinstance(ts, TimeseriesObject)
