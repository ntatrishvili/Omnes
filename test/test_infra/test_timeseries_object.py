"""
Test suite for TimeseriesObject class with xarray.DataArray backend.

This test module validates both backward compatibility with the pandas-based API
and new xarray-specific features for multi-dimensional energy system modeling.

Test Coverage:
- Basic initialization (empty, DataFrame, xarray, CSV)
- Backward compatibility methods (to_df, to_nd, get_values)
- Frequency handling and resampling
- xarray-specific features (selection, metadata, multi-dimensional data)
- Complex energy scenarios with multiple dimensions and coordinates
"""

import unittest

import numpy as np
import pandas as pd
import xarray as xr
import tempfile
import os

from app.infra.timeseries_object import TimeseriesObject


class TestTimeseriesObject(unittest.TestCase):
    def test_init_empty(self):
        ts = TimeseriesObject()
        self.assertTrue(ts.empty())
        self.assertIsNone(ts.freq)

    def test_init_with_dataframe(self):
        df = pd.DataFrame(
            {"val": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="h")
        )
        ts = TimeseriesObject(data=df)
        self.assertFalse(ts.empty())
        self.assertIsInstance(ts.data, xr.DataArray)

    def test_init_with_xarray(self):
        """Test initialization with xarray DataArray."""
        times = pd.date_range("2020-01-01", periods=4, freq="1h")
        data_array = xr.DataArray(
            data=[10, 20, 30, 40],
            dims=["time"],
            coords={"time": times},
            attrs={"technology_type": "pv", "location": "rooftop_A"},
        )
        ts = TimeseriesObject(data=data_array)
        self.assertFalse(ts.empty())
        self.assertEqual(ts.data.attrs["technology_type"], "pv")
        self.assertEqual(ts.data.attrs["location"], "rooftop_A")

    def test_init_with_metadata(self):
        """Test initialization with metadata attributes."""
        df = pd.DataFrame(
            {"power": [5, 10, 15]},
            index=pd.date_range("2020-01-01", periods=3, freq="1h"),
        )
        ts = TimeseriesObject(
            data=df, entity_id="pv1", technology_type="solar", location="building_A"
        )
        self.assertEqual(ts.get_metadata("entity_id"), "pv1")
        self.assertEqual(ts.get_metadata("technology_type"), "solar")
        self.assertEqual(ts.get_metadata("location"), "building_A")

    def test_normalize_freq(self):
        self.assertEqual(TimeseriesObject.normalize_freq("h"), "1h")
        self.assertEqual(TimeseriesObject.normalize_freq("15min"), "15min")

    def test_to_df_backward_compatibility(self):
        """Test to_df() method for backward compatibility."""
        df = pd.DataFrame(
            {"val": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="h")
        )
        ts = TimeseriesObject(data=df)
        result_df = ts.to_df()
        self.assertIsInstance(result_df, pd.DataFrame)
        np.testing.assert_array_equal(result_df.values.flatten(), [1, 2, 3])

    def test_read_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            TimeseriesObject.read("not_a_file.csv", "val")

    def test_read_file_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.close()
            try:
                with self.assertRaises(ValueError):
                    TimeseriesObject.read(tmp.name, "val")
            finally:
                os.remove(tmp.name)

    def test_read_file_col_not_found(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("timestamp;foo\n2020.01.01 00:00;1\n")
            tmp.close()
            try:
                with self.assertRaises(KeyError):
                    TimeseriesObject.read(tmp.name, "bar")
            finally:
                os.remove(tmp.name)


class TestTimeseriesObjectExtended(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"val": [1, 2, 3, 4]},
            index=pd.date_range("2020-01-01", periods=4, freq="1h"),
        )
        self.ts = TimeseriesObject(data=self.df)

    def test_to_1h(self):
        ts_1h = self.ts.to_1h()
        self.assertEqual(ts_1h.freq, "1h")
        # Compare the values, not exact DataFrame (might have different indices)
        np.testing.assert_array_equal(
            ts_1h.to_df().values.flatten(), self.df.values.flatten()
        )

    def test_to_15m(self):
        ts_15m = self.ts.to_15m()
        self.assertEqual(ts_15m.freq, "15min")
        # With xarray backend, check the data size
        self.assertEqual(
            ts_15m.data.sizes["time"], (len(self.df) - 1) * 4 + 1
        )  # interpolated

    def test_resample_to_interpolate(self):
        ts_interp = self.ts.resample_to("15min", method="interpolate")
        # Get the pandas DataFrame for detailed testing
        df_result = ts_interp.to_df()
        self.assertAlmostEqual(df_result.iloc[1].iloc[0], 1.25)

    def test_resample_to_ffill(self):
        ts_ffill = self.ts.resample_to("15min", method="ffill")
        df_result = ts_ffill.to_df()
        self.assertEqual(df_result.iloc[1].iloc[0], 1)

    def test_resample_to_bfill(self):
        ts_bfill = self.ts.resample_to("15min", method="bfill")
        df_result = ts_bfill.to_df()
        self.assertEqual(df_result.iloc[0].iloc[0], 1)
        self.assertEqual(df_result.iloc[1].iloc[0], 2)

    def test_resample_to_agg_sum(self):
        ts_down = self.ts.resample_to("2h", method="agg", agg="sum", closed="left")
        df_result = ts_down.to_df()
        values = df_result.iloc[:, 0].values
        np.testing.assert_array_equal(values, [3, 7])

    def test_to_nd(self):
        arr = self.ts.to_nd()
        np.testing.assert_array_equal(arr, np.array([1, 2, 3, 4]))

    def test_get_values_default(self):
        values = self.ts.get_values()
        np.testing.assert_array_equal(values, self.ts.to_nd())

    def test_get_values_custom_time_set(self):
        values = self.ts.get_values(time_set=2)
        self.assertEqual(len(values), 2)

    def test_get_values_different_freq(self):
        # First test with our existing 1h data
        original_len = len(self.ts.to_nd())
        self.assertEqual(original_len, 4)

        # Create a TimeSeries with different frequency to test resampling
        ts_15min = self.ts.resample_to("15min", method="interpolate")
        values_15min = ts_15min.get_values(freq="15min")
        # With interpolation, we should get more values: 4 hours * 4 intervals + 1 = 13
        expected_len = (len(self.df) - 1) * 4 + 1
        self.assertEqual(len(values_15min), expected_len)

    def test_eq_operator(self):
        other = TimeseriesObject(data=self.df.copy())
        self.assertEqual(self.ts, other)

        # Modify the xarray data values
        other.data.values[0] = 99
        self.assertNotEqual(self.ts, other)

    def test_getattr_delegation(self):
        # xarray DataArray has shape (4,) not (4, 1) like DataFrame
        self.assertEqual(self.ts.shape, (4,))
        with self.assertRaises(AttributeError):
            _ = self.ts.non_existent_attr

    def test_empty_true_and_false(self):
        empty_ts = TimeseriesObject()
        self.assertTrue(empty_ts.empty())
        self.assertFalse(self.ts.empty())


class TestXArrayFeatures(unittest.TestCase):
    """Test xarray-specific features and multi-dimensional capabilities."""

    def setUp(self):
        # Create multi-dimensional test data
        times = pd.date_range("2025-01-01", periods=24, freq="1h")
        rng = np.random.default_rng(42)  # Use modern numpy generator with seed
        self.multi_data = xr.DataArray(
            data=rng.random((24, 3, 2)),  # time x location x scenario
            dims=["time", "location", "scenario"],
            coords={
                "time": times,
                "location": ["site_A", "site_B", "site_C"],
                "scenario": ["optimistic", "pessimistic"],
            },
            attrs={"technology_type": "pv", "entity_id": "pv_farm_1"},
        )
        self.ts_multi = TimeseriesObject(data=self.multi_data)

        # Simple test data
        self.simple_df = pd.DataFrame(
            {"power": [10, 15, 20, 25]},
            index=pd.date_range("2025-01-01", periods=4, freq="1h"),
        )
        self.ts_simple = TimeseriesObject(data=self.simple_df)

    def test_selection_operations(self):
        """Test xarray-style selection operations."""
        # Select specific location
        site_a_ts = self.ts_multi.sel(location="site_A")
        self.assertNotIn("location", site_a_ts.data.dims)
        self.assertEqual(site_a_ts.data.sizes["scenario"], 2)

        # Select time range
        subset_ts = self.ts_multi.sel(
            time=slice("2025-01-01 06:00", "2025-01-01 12:00")
        )
        self.assertEqual(subset_ts.data.sizes["time"], 7)  # 6 hours + 1

    def test_integer_selection(self):
        """Test integer-based selection."""
        first_timestep = self.ts_multi.isel(time=0)
        self.assertNotIn("time", first_timestep.data.dims)

        first_three_times = self.ts_multi.isel(time=slice(0, 3))
        self.assertEqual(first_three_times.data.sizes["time"], 3)

    def test_add_dimension(self):
        """Test adding new dimensions."""
        expanded_ts = self.ts_simple.add_dimension("weather", ["sunny", "cloudy"])
        self.assertIn("weather", expanded_ts.data.dims)
        self.assertEqual(expanded_ts.data.sizes["weather"], 2)

    def test_metadata_operations(self):
        """Test metadata setting and getting."""
        # Set metadata
        ts_with_meta = self.ts_simple.set_metadata(
            peak_power=100, efficiency=0.95, constraints={"max_power": "< 80kW"}
        )

        # Get metadata
        self.assertEqual(ts_with_meta.get_metadata("peak_power"), 100)
        self.assertEqual(ts_with_meta.get_metadata("efficiency"), 0.95)

        # Get all metadata
        all_meta = ts_with_meta.get_metadata()
        self.assertIn("peak_power", all_meta)
        self.assertIn("constraints", all_meta)

    def test_multi_dimensional_backward_compatibility(self):
        """Test that multi-dimensional data still works with existing methods."""
        # to_nd() should flatten multi-dimensional data
        flat_array = self.ts_multi.to_nd()
        expected_size = 24 * 3 * 2  # time * location * scenario
        self.assertEqual(len(flat_array), expected_size)

        # get_values() should work
        values = self.ts_multi.get_values()
        self.assertEqual(len(values), expected_size)

    def test_complex_energy_scenario(self):
        """Test complex energy system modeling scenario."""
        # Create PV data with multiple scenarios
        times = pd.date_range("2025-01-01", periods=8, freq="1h")
        rng = np.random.default_rng(123)  # Use modern numpy generator with seed
        pv_data = xr.DataArray(
            data=rng.random((8, 2, 3)),  # time x weather x degradation_year
            dims=["time", "weather", "degradation_year"],
            coords={
                "time": times,
                "weather": ["sunny", "cloudy"],
                "degradation_year": [0, 1, 2],  # years of operation
            },
            attrs={
                "technology_type": "pv",
                "peak_power_kw": 100,
                "location": "rooftop_commercial",
            },
        )

        ts_pv = TimeseriesObject(data=pv_data)

        # Test aggregation across weather scenarios
        avg_weather = ts_pv.data.mean(dim="weather")
        self.assertEqual(avg_weather.dims, ("time", "degradation_year"))

        # Test selection of specific degradation year
        year_0 = ts_pv.sel(degradation_year=0)
        self.assertNotIn("degradation_year", year_0.data.dims)

        # Test backward compatibility methods still work
        df_export = ts_pv.to_df()
        self.assertIsInstance(df_export, pd.DataFrame)
        # For multi-dimensional data, DataFrame should have columns for each combination
        # Time dimension should be preserved as index
        self.assertEqual(len(df_export), 8)  # 8 time steps
        self.assertGreater(
            df_export.shape[1], 1
        )  # Multiple columns for multi-dimensional data


if __name__ == "__main__":
    unittest.main()
