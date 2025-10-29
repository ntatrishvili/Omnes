import unittest

from app.infra.timeseries_object_factory import TimeseriesFactory
from app.model.storage.battery import Battery


class DummyTimeseriesFactory(TimeseriesFactory):
    def create(self, name, **kwargs):
        return f"ts_{name}"


class TestBattery(unittest.TestCase):
    def setUp(self):
        self.ts_factory = DummyTimeseriesFactory()

    def test_battery_init(self):
        b = Battery(
            id="bat1",
            ts_factory=self.ts_factory,
            max_charge_rate=10,
            capacity=100,
            bus="bus1",
        )
        self.assertEqual(b.id, "bat1")
        self.assertEqual(b.max_charge_rate, 10)
        self.assertEqual(b.capacity, 100)
        self.assertEqual(b.bus, "bus1")
        self.assertIn("p_in", b.quantities)
        self.assertIn("p_out", b.quantities)
        self.assertIn("e_stor", b.quantities)
        self.assertEqual(str(b), "Battery 'bat1' with charging power=10 , capacity=100")


if __name__ == "__main__":
    unittest.main()
