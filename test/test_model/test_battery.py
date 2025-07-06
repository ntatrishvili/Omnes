import unittest
from app.model.battery import Battery
from app.model.timeseries_object_factory import TimeseriesFactory


class DummyTimeseriesFactory(TimeseriesFactory):
    def create(self, name, **kwargs):
        return f"ts_{name}"


class TestBattery(unittest.TestCase):
    def setUp(self):
        self.ts_factory = DummyTimeseriesFactory()

    def test_battery_init(self):
        b = Battery(id="bat1", ts_factory=self.ts_factory, max_power=10, capacity=100)
        self.assertEqual(b.id, "bat1")
        self.assertEqual(b.parameters["max_power"], 10)
        self.assertEqual(b.parameters["capacity"], 100)
        self.assertIn("p_bess_in", b.quantities)
        self.assertIn("p_bess_out", b.quantities)
        self.assertIn("e_bess_stor", b.quantities)
        self.assertEqual(str(b), "Battery 'bat1' with max_power=10, capacity=100")


if __name__ == "__main__":
    unittest.main()
