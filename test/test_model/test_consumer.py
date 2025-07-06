import unittest
import pandas as pd
from app.model.consumer import Consumer
from app.model.timeseries_object_factory import TimeseriesFactory

class DummyTimeseriesFactory(TimeseriesFactory):
    def create(self, name, **kwargs):
        return pd.Series([1, 2, 3])

class TestConsumer(unittest.TestCase):
    def setUp(self):
        self.ts_factory = DummyTimeseriesFactory()

    def test_consumer_init(self):
        c = Consumer(id="c1", ts_factory=self.ts_factory, max_power=7)
        self.assertEqual(c.id, "c1")
        self.assertIn("p_cons", c.quantities)
        self.assertIsInstance(c.quantities["p_cons"], pd.Series)
        self.assertIsInstance(str(c), str)

if __name__ == "__main__":
    unittest.main()
