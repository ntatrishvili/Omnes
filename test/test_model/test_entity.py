import unittest
from app.model.entity import Entity
from app.model.timeseries_object_factory import TimeseriesFactory


class DummyTimeseriesFactory(TimeseriesFactory):
    def create(self, name, **kwargs):
        return f"ts_{name}"


class TestEntity(unittest.TestCase):
    def setUp(self):
        self.ts_factory = DummyTimeseriesFactory()

    def test_entity_init(self):
        e = Entity(id="ent1", ts_factory=self.ts_factory)
        self.assertEqual(e.id, "ent1")
        self.assertEqual(e.ts_factory, self.ts_factory)
        self.assertIsInstance(e.parameters, dict)
        self.assertIsInstance(e.quantities, dict)
        self.assertIsInstance(str(e), str)

    def test_add_sub_entity(self):
        parent = Entity(id="parent", ts_factory=self.ts_factory)
        child = Entity(id="child", ts_factory=self.ts_factory)
        parent.add_sub_entity(child)
        self.assertIn(child, parent.sub_entities)
        self.assertEqual(child.parent, parent)
        self.assertEqual(child.parent_id, parent.id)

    def test_getitem(self):
        e = Entity(id="ent1", ts_factory=self.ts_factory)
        e.parameters["foo"] = 42
        e.quantities["bar"] = "baz"
        self.assertEqual(e.foo, 42)
        self.assertEqual(e.bar, "baz")
        with self.assertRaises(KeyError):
            _ = e.notfound


if __name__ == "__main__":
    unittest.main()
