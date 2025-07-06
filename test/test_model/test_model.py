import unittest
from unittest.mock import patch
from app.model.model import Model
from app.model.timeseries_object_factory import TimeseriesFactory


class DummyTimeseriesFactory(TimeseriesFactory):
    def create(self, name, **kwargs):
        # Return a minimal object with .empty and .sum()
        class DummyTS:
            empty = True

            def sum(self):
                return 0

        return DummyTS()


class TestModel(unittest.TestCase):
    def setUp(self):
        self.ts_factory = DummyTimeseriesFactory()

    def test_model_init(self):
        m = Model(id="m1", ts_factory=self.ts_factory)
        self.assertEqual(m.id, "m1")
        self.assertIsInstance(str(m), str)

    @patch("app.model.model.get_input_path", side_effect=lambda x: x)
    @patch("app.model.pv.PV.__init__", return_value=None)
    @patch("app.model.consumer.Consumer.__init__", return_value=None)
    @patch("app.model.battery.Battery.__init__", return_value=None)
    @patch("app.model.entity.Entity.add_sub_entity", return_value=None)
    @patch("app.model.model.Slack.__init__", return_value=None)
    def test_build_minimal(
        self,
        mock_slack_init,
        mock_add_sub_entity,
        mock_battery_init,
        mock_consumer_init,
        mock_pv_init,
        mock_get_input_path,
    ):
        config = {
            "entity1": {
                "pvs": {"pv1": {"filename": "dummy.csv"}},
                "consumers": {"c1": {"filename": "dummy.csv"}},
                "batteries": {"b1": {"nominal_power": 5, "capacity": 10}},
            }
        }
        m = Model.build(config, time_set=10, frequency="15min")
        self.assertIsInstance(m, Model)
        self.assertEqual(m.time_set, 10)
        self.assertEqual(m.frequency, "15min")
        # Should have 2 entities: entity1 and slack
        self.assertEqual(len(m.entities), 2)
        # Check that the correct calls were made
        mock_pv_init.assert_called()
        mock_consumer_init.assert_called()
        mock_battery_init.assert_called()
        mock_add_sub_entity.assert_called()
        mock_slack_init.assert_called()


if __name__ == "__main__":
    unittest.main()
