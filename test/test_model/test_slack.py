import unittest

import pandas as pd

from app.infra.quantity_factory import QuantityFactory
from app.model.slack import Slack


class DummyQuantityFactory(QuantityFactory):
    def create(self, name, **kwargs):
        return pd.Series([1, 2, 3])


class TestSlack(unittest.TestCase):
    def setUp(self):
        self.quantity_factory = DummyQuantityFactory()

    def test_slack_init(self):
        s = Slack(id="s1", quantity_factory=self.quantity_factory)
        self.assertEqual(s.id, "s1")
        self.assertIn("p_in", s.quantities)
        self.assertIn("p_out", s.quantities)
        self.assertIsInstance(s.p_in, pd.Series)
        self.assertIsInstance(s.p_out, pd.Series)
        self.assertIsInstance(str(s), str)


if __name__ == "__main__":
    unittest.main()
