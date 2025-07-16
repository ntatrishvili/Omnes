import unittest
import pandas as pd
from app.model.pv import PV
from app.model.timeseries_object_factory import TimeseriesFactory


class DummyTimeseriesFactory(TimeseriesFactory):
    def create(self, name, **kwargs):
        # Return a minimal pandas Series to satisfy .empty and .sum()
        return pd.Series([1, 2, 3])


class TestPV(unittest.TestCase):
    def setUp(self):
        self.ts_factory = DummyTimeseriesFactory()

    def test_pv_init(self):
        pv = PV(id="pv1", ts_factory=self.ts_factory, max_power=5)
        self.assertEqual(pv.id, "pv1")
        self.assertIn("p_pv", pv.quantities)
        self.assertIsInstance(pv.quantities["p_pv"], pd.Series)
        self.assertIsInstance(str(pv), str)


if __name__ == "__main__":
    unittest.main()
