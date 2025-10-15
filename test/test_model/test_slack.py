import unittest

import pandas as pd

from app.infra.timeseries_object_factory import TimeseriesFactory
from app.model.slack import Slack


class DummyTimeseriesFactory(TimeseriesFactory):
    def create(self, name, **kwargs):
        return pd.Series([1, 2, 3])


class TestSlack(unittest.TestCase):
    def setUp(self):
        self.ts_factory = DummyTimeseriesFactory()

    def test_slack_init(self):
        s = Slack(id="s1", ts_factory=self.ts_factory)
        self.assertEqual(s.id, "s1")
        self.assertIn("p_slack_in", s.quantities)
        self.assertIn("p_slack_out", s.quantities)
        self.assertIsInstance(s.p_slack_in, pd.Series)
        self.assertIsInstance(s.p_slack_out, pd.Series)
        self.assertIsInstance(str(s), str)


if __name__ == "__main__":
    unittest.main()
