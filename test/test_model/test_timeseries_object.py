import unittest
import pandas as pd
import tempfile
import os
from app.model.timeseries_object import TimeseriesObject

class TestTimeseriesObject(unittest.TestCase):
    def test_init_empty(self):
        ts = TimeseriesObject()
        self.assertTrue(ts.data.empty)
        self.assertIsNone(ts.freq)

    def test_init_with_dataframe(self):
        df = pd.DataFrame({"val": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="H"))
        ts = TimeseriesObject(data=df)
        self.assertFalse(ts.data.empty)
        self.assertIsInstance(ts.data, pd.DataFrame)

    def test_normalize_freq(self):
        self.assertEqual(TimeseriesObject.normalize_freq("H"), "1H")
        self.assertEqual(TimeseriesObject.normalize_freq("15min"), "15min")

    def test_to_df(self):
        df = pd.DataFrame({"val": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="H"))
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

if __name__ == "__main__":
    unittest.main()
