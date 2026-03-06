import unittest
from unittest.mock import patch

import pandas as pd
import pulp

import app.conversion.pulp_converter
from app.infra import util
from app.infra.util import TimeSet, TimesetBuilder


class TestUtil(unittest.TestCase):

    def test_create_empty_pulp_var(self):
        pulp_vars = app.conversion.pulp_converter.create_empty_pulp_var("foo", 3)
        self.assertEqual(len(pulp_vars), 3)
        for i, v in enumerate(pulp_vars):
            self.assertIsInstance(v, pulp.LpVariable)
            self.assertIn(f"foo_{i}", v.name)

    def test_flatten(self):
        nested = [[1, 2], [3, [4, 5]], 6]
        flat = util.flatten(nested)
        self.assertEqual(flat, [1, 2, 3, 4, 5, 6])

    def test_flatten_empty_list(self):
        self.assertEqual(util.flatten([]), [])

    def test_flatten_deeply_nested(self):
        nested = [[[1, [2, [3]]]]]
        self.assertEqual(util.flatten(nested), [1, 2, 3])

    def test_cast_like_none_sample_returns_value(self):
        self.assertEqual(util.cast_like("abc", None), "abc")

    def test_cast_like_numeric_conversions(self):
        # integer from string
        self.assertEqual(util.cast_like("42", 0), 42)
        # float from string
        self.assertAlmostEqual(util.cast_like("3.14", 0.0), 3.14)
        # float from string with comma
        self.assertAlmostEqual(util.cast_like("1,234.56", 0.0), 1234.56)

    def test_cast_like_bool_and_strings(self):
        # cast_like now only converts numeric types; for non-numeric samples it returns the original value
        self.assertEqual(util.cast_like("yes", True), "yes")
        self.assertEqual(util.cast_like("No", True), "No")
        # non-string to bool returns the original value as well
        self.assertEqual(util.cast_like(0, True), 0)

    def test_cast_like_sequence_and_mapping(self):
        # cast_like only converts numeric target types; non-numeric samples are returned unchanged
        self.assertEqual(util.cast_like("[1,2,3]", []), "[1,2,3]")
        self.assertEqual(util.cast_like('{"a":1}', {}), '{"a":1}')
        self.assertEqual(util.cast_like([1, 2], tuple()), [1, 2])
        self.assertEqual(util.cast_like((3, 4), list()), (3, 4))
        # string sample: non-numeric sample -> return original value
        self.assertEqual(util.cast_like(123, ""), 123)

    def test_cast_like_invalid_numeric_raises(self):
        with self.assertRaises(Exception):
            util.cast_like("not a number", 0)

    def test_cast_like_already_correct_type(self):
        """Test that values already of the correct type are returned as-is"""
        self.assertEqual(util.cast_like(42, 0), 42)
        self.assertAlmostEqual(util.cast_like(3.14, 0.0), 3.14)

    def test_timesetbuilder_create_periods_and_tz(self):
        ts = util.TimesetBuilder.create(
            number_of_time_steps=5, time_start="2020-01-01", resolution="1D", tz="UTC"
        )
        self.assertEqual(ts.number_of_time_steps, 5)
        self.assertEqual(ts.tz, "UTC")
        self.assertEqual(len(ts.time_points), 5)
        # resolution should be a pandas frequency-like object
        self.assertIsNotNone(ts.resolution)


class TestTimeSet(unittest.TestCase):
    """Tests for the TimeSet class"""

    def test_timeset_repr(self):
        """Test TimeSet string representation"""
        ts = TimesetBuilder.create(
            time_start="2024-01-01", resolution="1h", number_of_time_steps=24
        )
        repr_str = repr(ts)
        self.assertIn("TimeSet", repr_str)
        self.assertIn("start=", repr_str)
        self.assertIn("end=", repr_str)
        self.assertIn("resolution=", repr_str)
        self.assertIn("steps=", repr_str)

    def test_timeset_hex_id_unique(self):
        """Test that different TimeSets have different hex_ids"""
        ts1 = TimesetBuilder.create(
            time_start="2024-01-01", resolution="1h", number_of_time_steps=24
        )
        ts2 = TimesetBuilder.create(
            time_start="2024-01-02", resolution="1h", number_of_time_steps=24
        )
        self.assertNotEqual(ts1.hex_id, ts2.hex_id)

    def test_timeset_hex_id_same_for_identical(self):
        """Test that identical TimeSets have the same hex_id"""
        ts1 = TimesetBuilder.create(
            time_start="2024-01-01", resolution="1h", number_of_time_steps=24
        )
        ts2 = TimesetBuilder.create(
            time_start="2024-01-01", resolution="1h", number_of_time_steps=24
        )
        self.assertEqual(ts1.hex_id, ts2.hex_id)

    def test_timeset_hex_id_length(self):
        """Test that hex_id is 8 characters"""
        ts = TimesetBuilder.create(
            time_start="2024-01-01", resolution="1h", number_of_time_steps=10
        )
        self.assertEqual(len(ts.hex_id), 8)

    def test_timeset_freq_property(self):
        """Test the freq property returns proper frequency string with leading digit"""
        # Test with explicit leading digit
        ts = TimesetBuilder.create(
            time_start="2024-01-01", resolution="1h", number_of_time_steps=10
        )
        freq = ts.freq
        self.assertIsInstance(freq, str)
        self.assertTrue(freq[0].isdigit())

        # Test that leading 1 is added when no digit present
        ts2 = TimesetBuilder.create(
            time_start="2024-01-01", resolution="h", number_of_time_steps=10
        )
        self.assertTrue(ts2.freq.startswith("1"))

    def test_timeset_freq_none_resolution(self):
        """Test freq property when resolution is None"""
        ts = TimeSet(
            start="2024-01-01",
            end="2024-01-02",
            resolution=None,
            number_of_time_steps=24,
            time_points=pd.date_range("2024-01-01", periods=24, freq="h"),
        )
        self.assertEqual(ts.freq, "")

    def test_timeset_freq_with_pandas_offset(self):
        """Test freq property with a pandas offset object"""
        ts = TimesetBuilder.create(
            time_start="2024-01-01", resolution="15min", number_of_time_steps=96
        )
        freq = ts.freq
        self.assertIn("15", freq)

    def test_timeset_freq_with_string_resolution(self):
        """Test freq property with string resolution that has freqstr"""
        ts = TimesetBuilder.create(
            time_start="2024-01-01", resolution="D", number_of_time_steps=7
        )
        freq = ts.freq
        self.assertIsInstance(freq, str)
        self.assertTrue(len(freq) > 0)


class TestTimesetBuilder(unittest.TestCase):
    """Tests for TimesetBuilder.create method"""

    def test_create_with_start_end_freq(self):
        """Test creating TimeSet with start, end, and freq"""
        ts = TimesetBuilder.create(
            time_start="2024-01-01", time_end="2024-01-02", resolution="1h"
        )
        self.assertIsNotNone(ts)
        self.assertEqual(ts.start, "2024-01-01")

    def test_create_with_start_periods_freq(self):
        """Test creating TimeSet with start, periods, and freq"""
        ts = TimesetBuilder.create(
            time_start="2024-01-01", number_of_time_steps=48, resolution="1h"
        )
        self.assertEqual(ts.number_of_time_steps, 48)

    def test_create_with_all_four_params_drops_end(self):
        """Test that when all 4 params provided, end is dropped"""
        ts = TimesetBuilder.create(
            time_start="2024-01-01",
            time_end="2024-01-10",
            number_of_time_steps=24,
            resolution="1h",
        )
        # Should use start + periods + freq, ignoring end
        self.assertEqual(ts.number_of_time_steps, 24)

    def test_create_defaults_start_when_none_provided(self):
        """Test that start defaults to 1970-01-01 when neither start nor end provided"""
        ts = TimesetBuilder.create(number_of_time_steps=10, resolution="1h")
        self.assertEqual(ts.start, "1970-01-01")

    def test_create_defaults_freq_when_missing(self):
        """Test that freq defaults to hourly and periods to 10 when only start provided"""
        ts = TimesetBuilder.create(time_start="2024-01-01")
        # Should default freq to 'h' and periods to 10
        self.assertEqual(ts.number_of_time_steps, 10)

    def test_create_with_time_kwargs(self):
        """Test passing additional kwargs to pandas date_range"""
        ts = TimesetBuilder.create(
            time_start="2024-01-01",
            resolution="1h",
            number_of_time_steps=24,
            time_kwargs={"normalize": True},
        )
        self.assertIsNotNone(ts)

    def test_create_with_end_and_freq_only(self):
        """Test creating TimeSet with end and freq (defaults start)"""
        ts = TimesetBuilder.create(
            time_end="2024-01-02", resolution="1h", number_of_time_steps=24
        )
        self.assertIsNotNone(ts)
        self.assertEqual(ts.number_of_time_steps, 24)

    def test_create_various_resolutions(self):
        """Test creating TimeSet with various resolution formats"""
        resolutions = ["1h", "30min", "15min", "1D", "2h"]
        for res in resolutions:
            ts = TimesetBuilder.create(
                time_start="2024-01-01", resolution=res, number_of_time_steps=10
            )
            self.assertEqual(ts.number_of_time_steps, 10)


class TestGetInputPath(unittest.TestCase):
    """Tests for get_input_path function"""

    @patch("os.path.isdir")
    def test_get_input_path_folder_not_found(self, mock_isdir):
        """Test that FileNotFoundError is raised when data folder doesn't exist"""
        mock_isdir.return_value = False
        with self.assertRaises(FileNotFoundError) as context:
            util.get_input_path("test.csv")
        self.assertIn("data folder", str(context.exception))

    @patch("os.path.isfile")
    @patch("os.path.isdir")
    def test_get_input_path_file_not_found_message(self, mock_isdir, mock_isfile):
        """Test the error message when file is not found"""
        mock_isdir.return_value = True
        mock_isfile.return_value = False
        with self.assertRaises(FileNotFoundError) as context:
            util.get_input_path("missing.csv")
        self.assertIn("missing.csv", str(context.exception))


if __name__ == "__main__":
    unittest.main()
