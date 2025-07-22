import unittest

import numpy as np
import pandas as pd
import tempfile
import os

from app.infra.timeseries_object import TimeseriesObject


class TestTimeseriesObject(unittest.TestCase):
    def test_init_empty(self):
        ts = TimeseriesObject()
        self.assertTrue(ts.data.empty)
        self.assertIsNone(ts.freq)

    def test_init_with_dataframe(self):
        df = pd.DataFrame(
            {"val": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="h")
        )
        ts = TimeseriesObject(data=df)
        self.assertFalse(ts.data.empty)
        self.assertIsInstance(ts.data, pd.DataFrame)

    def test_normalize_freq(self):
        self.assertEqual(TimeseriesObject.normalize_freq("h"), "1h")
        self.assertEqual(TimeseriesObject.normalize_freq("15min"), "15min")

    def test_to_df(self):
        df = pd.DataFrame(
            {"val": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="h")
        )
        ts = TimeseriesObject(data=df)
        pd.testing.assert_frame_equal(ts.to_df(), df)

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
        pd.testing.assert_frame_equal(ts_1h.to_df(), self.df)

    def test_to_15m(self):
        ts_15m = self.ts.to_15m()
        self.assertEqual(ts_15m.freq, "15min")
        self.assertEqual(len(ts_15m.data), (len(self.df) - 1) * 4 + 1)  # interpolated

    def test_resample_to_interpolate(self):
        ts_interp = self.ts.resample_to("15min", method="interpolate")
        self.assertAlmostEqual(ts_interp.data.iloc[1].val, 1.25)

    def test_resample_to_ffill(self):
        ts_ffill = self.ts.resample_to("15min", method="ffill")
        self.assertEqual(ts_ffill.data.iloc[1].val, 1)

    def test_resample_to_bfill(self):
        ts_bfill = self.ts.resample_to("15min", method="bfill")
        self.assertEqual(ts_bfill.data.iloc[0].val, 1)
        self.assertEqual(ts_bfill.data.iloc[1].val, 2)

    def test_resample_to_agg_sum(self):
        ts_down = self.ts.resample_to("2h", method="agg", agg="sum", closed="left")
        self.assertTrue((ts_down.data["val"] == [3, 7]).all())

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
        values = self.ts.get_values(freq="15min")
        self.assertGreater(len(values), len(self.ts.to_nd()))

    def test_eq_operator(self):
        other = TimeseriesObject(data=self.df.copy())
        self.assertTrue(self.ts == other)

        other.data.iloc[0] = 99
        self.assertFalse(self.ts == other)

    def test_getattr_delegation(self):
        self.assertEqual(self.ts.shape, (4, 1))
        with self.assertRaises(AttributeError):
            _ = self.ts.non_existent_attr

    def test_empty_true_and_false(self):
        empty_ts = TimeseriesObject()
        self.assertTrue(empty_ts.empty())
        self.assertFalse(self.ts.empty())


if __name__ == "__main__":
    unittest.main()
