import unittest

from app.infra.timeseries_object_factory import TimeseriesFactory
from app.model.storage.hot_water_storage import HotWaterStorage


class DummyTimeseriesFactory(TimeseriesFactory):
    def create(self, name, **kwargs):
        return f"ts_{name}"


class TestHotWaterStorage(unittest.TestCase):
    def setUp(self):
        self.ts_factory = DummyTimeseriesFactory()

    def test_hot_water_storage_init(self):
        hw = HotWaterStorage(
            id="hw1",
            ts_factory=self.ts_factory,
            volume=150,
            set_temperature=60,
        )
        self.assertEqual(hw.id, "hw1")
        self.assertIn("volume", hw.quantities)
        self.assertIn("set_temperature", hw.quantities)
        self.assertEqual(
            str(hw),
            "Hot water storage 'hw1' with volume: 150 l and set temperature 60 \u00b0C",
        )


if __name__ == "__main__":
    unittest.main()

