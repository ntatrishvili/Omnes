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


def test_resample_to_upsampling():
    """Test upsampling from hourly to 15-minute frequency"""
    df = pd.DataFrame(
        {"power": [10, 20, 30, 40]},
        index=pd.date_range("2025-01-01", periods=4, freq="1h"),
    )
    ts = TimeseriesObject(data=df, freq="1h")
    ts_15min = ts.resample_to("15min")
    assert ts_15min.freq == "15min"
    assert ts_15min.data.sizes["timestamp"] == 16  # 4 hours * 4 = 16 intervals


def test_resample_to_downsampling():
    """Test downsampling from 15-minute to hourly frequency"""
    df = pd.DataFrame(
        {"power": [10, 15, 20, 25] * 4},
        index=pd.date_range("2025-01-01", periods=16, freq="15min"),
    )
    ts = TimeseriesObject(data=df, freq="15min")
    ts_1h = ts.resample_to("1h", agg="sum")
    assert ts_1h.freq == "1h"
    assert ts_1h.data.sizes["timestamp"] == 4


def test_resample_to_in_place():
    """Test in-place resampling"""
    df = pd.DataFrame(
        {"power": [10, 20, 30]},
        index=pd.date_range("2025-01-01", periods=3, freq="1h"),
    )
    ts = TimeseriesObject(data=df, freq="1h")
    original_id = id(ts)
    ts.resample_to("30min", in_place=True)
    assert id(ts) == original_id
    assert ts.freq == "30min"


def test_resample_to_same_freq():
    """Test resampling to same frequency returns self"""
    df = pd.DataFrame(
        {"power": [10, 20]}, index=pd.date_range("2025-01-01", periods=2, freq="1h")
    )
    ts = TimeseriesObject(data=df, freq="1h")
    ts_same = ts.resample_to("1h")
    assert ts_same is ts


def test_to_1h_conversion():
    """Test convenience method to_1h"""
    df = pd.DataFrame(
        {"power": [10, 15, 20, 25]},
        index=pd.date_range("2025-01-01", periods=4, freq="15min"),
    )
    ts = TimeseriesObject(data=df, freq="15min")
    ts_hourly = ts.to_1h()
    assert ts_hourly.freq == "1h"
    assert ts_hourly.data.sizes["timestamp"] == 1


def test_to_15m_conversion():
    """Test convenience method to_15m"""
    df = pd.DataFrame(
        {"power": [10, 20]}, index=pd.date_range("2025-01-01", periods=2, freq="1h")
    )
    ts = TimeseriesObject(data=df, freq="1h")
    ts_15min = ts.to_15m()
    assert ts_15min.freq == "15min"
    assert ts_15min.data.sizes["timestamp"] == 8


def test_to_1h_raises_without_freq():
    """Test to_1h raises error when freq not set"""
    ts = TimeseriesObject()
    with pytest.raises(ValueError, match="Frequency of the time series is not set"):
        ts.to_1h()


def test_value_with_resampling():
    """Test value() method with frequency conversion"""
    df = pd.DataFrame(
        {"power": [10, 20, 30, 40]},
        index=pd.date_range("2025-01-01", periods=4, freq="1h"),
    )
    ts = TimeseriesObject(data=df, freq="1h")
    # Get values at different frequency
    values_15min = ts.value(freq="15min")
    assert len(values_15min) == 16


def test_value_with_time_set_slicing():
    """Test value() method with time_set parameter"""
    df = pd.DataFrame(
        {"power": [10, 20, 30, 40, 50]},
        index=pd.date_range("2025-01-01", periods=5, freq="1h"),
    )
    ts = TimeseriesObject(data=df, freq="1h")
    values_subset = ts.value(time_set=3)
    assert len(values_subset) == 3
    np.testing.assert_array_equal(values_subset, [10, 20, 30])


def test_read_csv_with_custom_time_column():
    """Test reading CSV with custom datetime column name"""
    import tempfile
    import os

    csv_content = """date,power
2025-01-01 00:00:00,10
2025-01-01 01:00:00,20
2025-01-01 02:00:00,30"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = f.name

    try:
        ts = TimeseriesObject.read(temp_path, col="power", time_col="date")
        assert not ts.empty()
        assert ts.data.sizes["timestamp"] == 3
    finally:
        os.unlink(temp_path)


def test_read_csv_with_datetime_format():
    """Test reading CSV with custom datetime format"""
    import tempfile
    import os

    csv_content = """timestamp,value
01/01/2025 12:00,100
01/01/2025 13:00,200"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = f.name

    try:
        ts = TimeseriesObject.read(
            temp_path,
            col="value",
            time_col="timestamp",
            datetime_format="%d/%m/%Y %H:%M",
        )
        assert not ts.empty()
        assert ts.data.sizes["timestamp"] == 2
    finally:
        os.unlink(temp_path)


def test_init_with_iterable_and_coords():
    """Test initialization with iterable data and coordinates"""
    coords = pd.date_range("2025-01-01", periods=3, freq="1h")
    ts = TimeseriesObject(data=[10, 20, 30], coords=coords)
    assert ts.data.sizes["timestamp"] == 3
    assert ts.data.values[0] == 10


def test_multi_column_dataframe_to_xarray():
    """Test conversion of multi-column DataFrame"""
    df = pd.DataFrame(
        {"col1": [1, 2, 3], "col2": [4, 5, 6]},
        index=pd.date_range("2025-01-01", periods=3, freq="1h"),
    )
    ts = TimeseriesObject(data=df)
    assert "variable" in ts.data.dims
    assert ts.data.sizes["variable"] == 2


def test_to_df_multi_dimensional():
    """Test DataFrame conversion for multi-dimensional data"""
    times = pd.date_range("2025-01-01", periods=3, freq="1h")
    data = xr.DataArray(
        data=np.random.rand(3, 2),
        dims=["timestamp", "scenario"],
        coords={"timestamp": times, "scenario": ["A", "B"]},
    )
    ts = TimeseriesObject(data=data)
    df = ts.to_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df.shape[1] == 2


def test_equality():
    """Test equality comparison between TimeseriesObject instances"""
    df1 = pd.DataFrame(
        {"power": [10, 20]}, index=pd.date_range("2025-01-01", periods=2, freq="1h")
    )
    ts1 = TimeseriesObject(data=df1, freq="1h")
    ts2 = TimeseriesObject(data=df1.copy(), freq="1h")
    ts3 = TimeseriesObject(data=df1 * 2, freq="1h")

    assert ts1 == ts2
    assert not (ts1 == ts3)
    assert not (ts1 == "not a timeseries")


def test_repr():
    """Test string representation"""
    df = pd.DataFrame(
        {"power": [10, 20, 30]},
        index=pd.date_range("2025-01-01", periods=3, freq="1h"),
    )
    ts = TimeseriesObject(data=df, freq="1h")
    repr_str = repr(ts)
    assert "TimeseriesObject" in repr_str
    assert "freq='1h'" in repr_str


def test_set_method():
    """Test set() method for updating data"""
    df1 = pd.DataFrame(
        {"power": [10, 20]}, index=pd.date_range("2025-01-01", periods=2, freq="1h")
    )
    ts = TimeseriesObject(data=df1, freq="1h")

    df2 = pd.DataFrame(
        {"power": [30, 40, 50]},
        index=pd.date_range("2025-01-02", periods=3, freq="1h"),
    )
    ts.set(df2, freq="1h")
    assert ts.data.sizes["timestamp"] == 3
    assert ts.data.values[0] == 30


def test_resample_invalid_method():
    """Test resampling with invalid method raises error"""
    df = pd.DataFrame(
        {"power": [10, 20]}, index=pd.date_range("2025-01-01", periods=2, freq="1h")
    )
    ts = TimeseriesObject(data=df, freq="1h")
    with pytest.raises(ValueError, match="Unsupported method"):
        ts.resample_to("30min", method="invalid_method")


def test_read_csv_missing_column():
    """Test reading CSV with missing column raises KeyError"""
    import tempfile
    import os

    csv_content = """timestamp,power
2025-01-01 00:00:00,10"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = f.name

    try:
        with pytest.raises(KeyError, match="missing_col"):
            TimeseriesObject.read(temp_path, col="missing_col")
    finally:
        os.unlink(temp_path)


def test_read_csv_missing_file():
    """Test reading non-existent CSV raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        TimeseriesObject.read("/nonexistent/path.csv", col="power")


def test_getattr_delegation():
    """Test __getattr__ delegates to underlying DataArray"""
    df = pd.DataFrame(
        {"power": [10, 20, 30]},
        index=pd.date_range("2025-01-01", periods=3, freq="1h"),
    )
    ts = TimeseriesObject(data=df, freq="1h")
    # Should delegate to DataArray
    assert hasattr(ts, "dims")
    assert hasattr(ts, "coords")


def test_getattr_raises_for_missing():
    """Test __getattr__ raises AttributeError for missing attributes"""
    ts = TimeseriesObject()
    with pytest.raises(AttributeError):
        _ = ts.nonexistent_attribute


def test_infer_freq_from_data():
    """Test frequency inference from data"""
    df = pd.DataFrame(
        {"power": [10, 20, 30]},
        index=pd.date_range("2025-01-01", periods=3, freq="30min"),
    )
    ts = TimeseriesObject(data=df)
    assert ts.freq == "30min"


def test_resample_preserves_metadata():
    """Test that resampling preserves metadata attributes"""
    df = pd.DataFrame(
        {"power": [10, 20, 30, 40]},
        index=pd.date_range("2025-01-01", periods=4, freq="1h"),
    )
    ts = TimeseriesObject(data=df, freq="1h", entity_id="pv1", location="roof")
    ts_resampled = ts.resample_to("30min")
    assert ts_resampled.get_metadata("entity_id") == "pv1"
    assert ts_resampled.get_metadata("location") == "roof"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
