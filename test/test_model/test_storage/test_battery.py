import unittest

from app.infra.parameter import Parameter
from app.infra.quantity_factory import QuantityFactory
from app.model.storage.battery import Battery


class DummyQuantityFactory(QuantityFactory):
    def create(self, name, **kwargs):
        return Parameter(value=kwargs.get("input", 0))


class TestBattery(unittest.TestCase):
    def setUp(self):
        self.quantity_factory = DummyQuantityFactory()

    def test_battery_init(self):
        b = Battery(
            id="bat1",
            quantity_factory=self.quantity_factory,
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
