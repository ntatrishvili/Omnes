import unittest

import pandas as pd

from app.infra.quantity_factory import QuantityFactory
from app.model.generator.pv import PV


class DummyQuantityFactory(QuantityFactory):
    def create(self, name, **kwargs):
        # Return a minimal pandas Series to satisfy .empty and .sum()
        return pd.Series([1, 2, 3])


class TestPV(unittest.TestCase):
    def setUp(self):
        self.quantity_factory = DummyQuantityFactory()

    def test_pv_init(self):
        pv = PV(
            id="pv1", quantity_factory=self.quantity_factory, max_power=5, bus="pv1"
        )
        self.assertEqual(pv.id, "pv1")
        self.assertIn("p_out", pv.quantities)
        self.assertIsInstance(pv.quantities["p_out"], pd.Series)
        self.assertIsInstance(str(pv), str)


if __name__ == "__main__":
    unittest.main()
