import unittest

import pandas as pd

from app.infra.quantity_factory import QuantityFactory
from app.model.load.load import Load


class DummyQuantityFactory(QuantityFactory):
    def create(self, name, **kwargs):
        return pd.Series([1, 2, 3])


class TestConsumer(unittest.TestCase):
    def setUp(self):
        self.quantity_factory = DummyQuantityFactory()

    def test_consumer_init(self):
        c = Load(id="c1", quantity_factory=self.quantity_factory, max_power=7, bus="c1")
        self.assertEqual(c.id, "c1")
        self.assertIn("p_cons", c.quantities)
        self.assertIsInstance(c.p_cons, pd.Series)
        self.assertIsInstance(str(c), str)


if __name__ == "__main__":
    unittest.main()
