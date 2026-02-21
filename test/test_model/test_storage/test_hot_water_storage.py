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
        # Check that __str__ contains key information
        hw_str = str(hw)
        self.assertIn("hw1", hw_str)
        self.assertIn("150", hw_str)
        self.assertIn("60", hw_str)

    def test_hot_water_storage_capacity_calculation(self):
        """Test that capacity is calculated correctly from volume and temperatures."""
        # Create storage with 150L volume, 60°C set temp, 12°C inlet (default)
        hw = HotWaterStorage(
            id="hw_calc",
            quantity_factory=self.quantity_factory,
            volume=150,  # liters
            set_temperature=60,  # °C
            water_in_temperature=12,  # °C
        )

        # Expected capacity: (60 - 12) * (4.186 / 3600) * 150 = 8.372 kWh
        # c_water = 4.186 / 3600 ≈ 0.001163 kWh/(L·K)
        expected_capacity = (60 - 12) * (4.186 / 3600) * 150

        capacity_param = hw.quantities.get("capacity")
        self.assertIsNotNone(capacity_param)
        actual_capacity = (
            capacity_param.value if hasattr(capacity_param, "value") else capacity_param
        )
        self.assertAlmostEqual(actual_capacity, expected_capacity, places=4)

    def test_hot_water_storage_charge_discharge_rates(self):
        """Test that max_charge_rate and max_discharge_rate are calculated correctly."""
        hw = HotWaterStorage(
            id="hw_rates",
            quantity_factory=self.quantity_factory,
            volume=150,
            set_temperature=60,
            water_in_temperature=12,
            time_to_charge=2.0,  # 2 hours to charge
            time_to_discharge=1.5,  # 1.5 hours to discharge
        )

        expected_capacity = (60 - 12) * (4.186 / 3600) * 150
        expected_charge_rate = expected_capacity / 2.0
        expected_discharge_rate = expected_capacity / 1.5

        charge_rate = hw.quantities.get("max_charge_rate")
        discharge_rate = hw.quantities.get("max_discharge_rate")

        self.assertIsNotNone(charge_rate)
        self.assertIsNotNone(discharge_rate)

        charge_val = charge_rate.value if hasattr(charge_rate, "value") else charge_rate
        discharge_val = (
            discharge_rate.value if hasattr(discharge_rate, "value") else discharge_rate
        )

        self.assertAlmostEqual(charge_val, expected_charge_rate, places=4)
        self.assertAlmostEqual(discharge_val, expected_discharge_rate, places=4)

    def test_hot_water_storage_explicit_capacity_override(self):
        """Test that explicitly provided capacity overrides calculation."""
        hw = HotWaterStorage(
            id="hw_override",
            quantity_factory=self.quantity_factory,
            volume=150,
            set_temperature=60,
            capacity=10.0,  # Explicit capacity override
        )

        capacity_param = hw.quantities.get("capacity")
        actual_capacity = (
            capacity_param.value if hasattr(capacity_param, "value") else capacity_param
        )
        self.assertEqual(actual_capacity, 10.0)


if __name__ == "__main__":
    unittest.main()
