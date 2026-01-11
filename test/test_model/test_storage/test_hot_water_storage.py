import unittest

from app.infra.parameter import Parameter
from app.infra.quantity_factory import QuantityFactory

from app.model.storage.hot_water_storage import HotWaterStorage


class DummyQuantityFactory(QuantityFactory):
    def create(self, name, **kwargs):
        return Parameter(value=kwargs.get("input", 0))


class TestHotWaterStorage(unittest.TestCase):
    def setUp(self):
        self.quantity_factory = DummyQuantityFactory()

    def test_hot_water_storage_init(self):
        hw = HotWaterStorage(
            id="hw1",
            quantity_factory=self.quantity_factory,
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
