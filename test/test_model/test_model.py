import unittest
from unittest.mock import patch

from app.infra.util import TimesetBuilder
from app.model.model import Model


class DummyTimesetBuilder(TimesetBuilder):
    def create(self, time_kwargs=None, **kwargs): ...


class TestModel(unittest.TestCase):
    def setUp(self):
        self.ts_builder = DummyTimesetBuilder()

    def test_model_init(self):
        m = Model(id="m1", timeset_builder=self.ts_builder)
        self.assertEqual(m.id, "m1")
        self.assertIsInstance(str(m), str)

    def test_model_init_without_id(self):
        """Test model initialization generates random ID when not provided"""
        m = Model(timeset_builder=self.ts_builder)
        self.assertIsNotNone(m.id)
        self.assertIsInstance(m.id, str)
        # Should be a hex string
        self.assertEqual(len(m.id), 32)

    def test_model_with_entities_list(self):
        """Test model initialization with entities list"""
        from app.model.entity import Entity

        e1 = Entity(id="entity1")
        e2 = Entity(id="entity2")
        m = Model(timeset_builder=self.ts_builder, entities=[e1, e2])
        self.assertEqual(len(m.entities), 2)
        self.assertIn("entity1", m.entities)
        self.assertIn("entity2", m.entities)

    def test_add_entity(self):
        """Test adding entity to model"""
        from app.model.entity import Entity

        m = Model(id="test_model", timeset_builder=self.ts_builder)
        e = Entity(id="new_entity")
        m.add_entity(e)
        self.assertIn("new_entity", m.entities)
        self.assertEqual(m.entities["new_entity"], e)

    def test_getitem_direct_entity(self):
        """Test accessing entity directly by ID"""
        from app.model.entity import Entity

        e = Entity(id="entity1")
        m = Model(timeset_builder=self.ts_builder, entities=[e])
        retrieved = m["entity1"]
        self.assertEqual(retrieved, e)

    def test_getitem_nested_entity(self):
        """Test accessing nested sub-entity"""
        from app.model.entity import Entity
        from app.model.device import Device

        sub_device = Device(id="device1")
        parent = Entity(id="parent")
        parent.add_sub_entity(sub_device)

        m = Model(timeset_builder=self.ts_builder, entities=[parent])
        retrieved = m["device1"]
        self.assertEqual(retrieved, sub_device)

    def test_getitem_raises_keyerror(self):
        """Test KeyError raised for non-existent entity"""
        m = Model(id="test", timeset_builder=self.ts_builder)
        with self.assertRaises(KeyError) as ctx:
            _ = m["nonexistent"]
        self.assertIn("nonexistent", str(ctx.exception))

    def test_set_method_with_entity(self):
        """Test set() method adds entity"""
        from app.model.entity import Entity

        m = Model(id="test", timeset_builder=self.ts_builder)
        e = Entity(id="new_entity")
        m.set({"new_entity": e})
        self.assertIn("new_entity", m.entities)

    def test_set_method_with_quantity(self):
        """Test set() method updates entity quantity"""
        from app.model.entity import Entity
        from app.infra.quantity import Parameter
        from app.infra.util import TimeSet
        import pandas as pd

        # Create a mock timeset
        m = Model(id="test", timeset_builder=self.ts_builder)
        m.time_set = TimeSet(
            start="2025-01-01",
            end="2025-01-02",
            resolution="1h",
            number_of_time_steps=24,
            time_points=pd.date_range("2025-01-01", periods=24, freq="1h"),
            tz=None,
        )

        e = Entity(id="entity1")
        e.quantities["param1"] = Parameter(value=10)
        m.add_entity(e)

        # Update the quantity
        m.set({"entity1.param1": 20})
        self.assertEqual(e.quantities["param1"].value, 20)

    def test_number_of_time_steps_property(self):
        """Test number_of_time_steps property delegates to time_set"""
        from app.infra.util import TimeSet
        import pandas as pd

        m = Model(id="test", timeset_builder=self.ts_builder)
        m.time_set = TimeSet(
            start="2025-01-01",
            end="2025-01-02",
            resolution="1h",
            number_of_time_steps=48,
            time_points=pd.date_range("2025-01-01", periods=48, freq="30min"),
            tz=None,
        )
        self.assertEqual(m.number_of_time_steps, 48)

    def test_frequency_property(self):
        """Test frequency property delegates to time_set"""
        from app.infra.util import TimeSet
        import pandas as pd

        m = Model(id="test", timeset_builder=self.ts_builder)
        m.time_set = TimeSet(
            start="2025-01-01",
            end="2025-01-02",
            resolution="15min",
            number_of_time_steps=96,
            time_points=pd.date_range("2025-01-01", periods=96, freq="15min"),
            tz=None,
        )
        self.assertEqual(m.frequency, "15min")

    def test_frequency_setter(self):
        """Test frequency setter updates time_set"""
        from app.infra.util import TimeSet
        import pandas as pd

        m = Model(id="test", timeset_builder=self.ts_builder)
        m.time_set = TimeSet(
            start="2025-01-01",
            end="2025-01-02",
            resolution="1h",
            number_of_time_steps=24,
            time_points=pd.date_range("2025-01-01", periods=24, freq="1h"),
            tz=None,
        )
        m.frequency = "30min"
        self.assertEqual(m.time_set.resolution, "30min")

    def test_convert_method(self):
        """Test convert() method delegates to converter"""
        m = Model(id="test", timeset_builder=self.ts_builder)

        mock_converter = unittest.mock.Mock()
        mock_converter.convert_model.return_value = {"result": "converted"}

        result = m.convert(mock_converter, time_set=100, new_freq="1h")

        mock_converter.convert_model.assert_called_once_with(
            m, time_set=100, new_freq="1h"
        )
        self.assertEqual(result, {"result": "converted"})

    def test_str_representation(self):
        """Test string representation includes all entities"""
        from app.model.entity import Entity

        e1 = Entity(id="entity1")
        e2 = Entity(id="entity2")
        m = Model(timeset_builder=self.ts_builder, entities=[e1, e2])

        str_repr = str(m)
        self.assertIsInstance(str_repr, str)
        # Should contain entity representations
        self.assertIn("entity1", str_repr)
        self.assertIn("entity2", str_repr)

    @patch("app.model.model.get_input_path", side_effect=lambda x: x)
    @patch("app.model.generator.pv.PV.__init__", return_value=None)
    @patch("app.model.load.load.Load.__init__", return_value=None)
    @patch("app.model.storage.battery.Battery.__init__", return_value=None)
    @patch("app.model.entity.Entity.add_sub_entity", return_value=None)
    @patch("app.model.entity.Entity.__getattr__", return_value=None)
    @patch("app.model.model.Slack.__init__", return_value=None)
    def test_build_minimal(
        self,
        mock_slack_init,
        mock_entity_getattr,
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
        m = Model.build("model", config, time_set=10, frequency="15min")
        self.assertIsInstance(m, Model)
        self.assertEqual(m.time_set.number_of_time_steps, 10)
        self.assertEqual(m.frequency, "15min")
        # Should have 2 entities: entity1 and slack
        self.assertEqual(len(m.entities), 2)
        # Check that the correct calls were made
        mock_pv_init.assert_called()
        mock_consumer_init.assert_called()
        mock_battery_init.assert_called()
        mock_add_sub_entity.assert_called()
        mock_slack_init.assert_called()

    @patch("app.model.model.get_input_path", side_effect=lambda x: x)
    @patch("app.model.generator.pv.PV.__init__", return_value=None)
    @patch("app.model.load.load.Load.__init__", return_value=None)
    @patch("app.model.entity.Entity.add_sub_entity", return_value=None)
    def test_build_without_batteries(
        self,
        mock_add_sub_entity,
        mock_consumer_init,
        mock_pv_init,
        mock_get_input_path,
    ):
        """Test build() without batteries in config"""
        config = {
            "entity1": {
                "pvs": {"pv1": {"filename": "pv.csv"}},
                "consumers": {"c1": {"filename": "load.csv"}},
                "batteries": {},
                "slacks": {"slack1": {}},
            }
        }
        m = Model.build("model", config, time_set=24, frequency="1h")
        self.assertIsInstance(m, Model)
        self.assertEqual(len(m.entities), 2)  # entity1 and slack

    def test_model_with_time_kwargs(self):
        """Test model initialization with time_kwargs"""
        from unittest.mock import Mock

        mock_builder = Mock()
        mock_timeset = Mock()
        mock_timeset.number_of_time_steps = 10
        mock_timeset.resolution = "1h"
        mock_builder.create.return_value = mock_timeset

        time_kwargs = {"tz": "UTC", "normalize": True}
        m = Model(
            id="test",
            timeset_builder=mock_builder,
            time_kwargs=time_kwargs,
            time_start="2025-01-01",
            time_end="2025-01-02",
        )

        # Verify time_kwargs was passed to builder
        mock_builder.create.assert_called_once()
        call_args = mock_builder.create.call_args
        # accept both positional or keyword passing of time_kwargs
        if call_args[0]:
            # first positional argument should be time_kwargs when positional was used
            self.assertEqual(call_args[0][0], time_kwargs)
        else:
            # otherwise expect it passed as a keyword
            self.assertIn("time_kwargs", call_args[1])
            self.assertEqual(call_args[1]["time_kwargs"], time_kwargs)

    def test_getitem_deeply_nested_entity(self):
        """Test accessing an entity nested multiple levels deep"""
        from app.model.entity import Entity
        from app.model.device import Device

        grandchild = Device(id="granddevice")
        child = Entity(id="child")
        child.add_sub_entity(grandchild)
        parent = Entity(id="parent")
        parent.add_sub_entity(child)

        m = Model(timeset_builder=self.ts_builder, entities=[parent])
        retrieved = m["granddevice"]
        self.assertEqual(retrieved, grandchild)

    def test__find_in_subentities_returns_none(self):
        """Directly test _find_in_subentities returns None when not found"""
        from app.model.entity import Entity

        parent = Entity(id="parent")
        # no sub-entities added
        m = Model(timeset_builder=self.ts_builder, entities=[parent])
        result = m._find_in_subentities(parent, "missing")
        self.assertIsNone(result)

    def test_getitem_keyerror_includes_model_id_and_entity_id(self):
        """Ensure KeyError message contains both the missing id and model id"""
        m = Model(id="MyModel", timeset_builder=self.ts_builder)
        missing_id = "no_such_entity"
        with self.assertRaises(KeyError) as ctx:
            _ = m[missing_id]
        msg = str(ctx.exception)
        self.assertIn(missing_id, msg)
        self.assertIn("MyModel", msg)


if __name__ == "__main__":
    unittest.main()
