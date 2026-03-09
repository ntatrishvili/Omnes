import unittest

from app.infra.quantity_factory import QuantityFactory
from app.model.entity import Entity


class DummyQuantityFactory(QuantityFactory):
    def create(self, name, **kwargs):
        return f"ts_{name}"


class TestEntity(unittest.TestCase):
    def setUp(self):
        self.quantity_factory = DummyQuantityFactory()

    def test_entity_init(self):
        e = Entity(id="ent1", quantity_factory=self.quantity_factory)
        self.assertEqual(e.id, "ent1")
        self.assertEqual(e.quantity_factory, self.quantity_factory)
        self.assertIsInstance(e.quantities, dict)
        self.assertIsInstance(str(e), str)

    def test_add_sub_entity(self):
        parent = Entity(id="parent", quantity_factory=self.quantity_factory)
        child = Entity(id="child", quantity_factory=self.quantity_factory)
        parent.add_sub_entity(child)
        self.assertIn("child", parent.sub_entities)
        self.assertEqual(child.parent, parent)
        self.assertEqual(child.parent_id, parent.id)

    def test_getitem(self):
        e = Entity(id="ent1", quantity_factory=self.quantity_factory)
        e.quantities["bar"] = "baz"
        e.quantities["foo"] = 42
        self.assertEqual(e.foo, 42)
        self.assertEqual(e.bar, "baz")
        with self.assertRaises(AttributeError):
            _ = e.notfound

    def test_contains(self):
        e = Entity(id="ent1", quantity_factory=self.quantity_factory)
        e.quantities["foo"] = 42
        self.assertIn("foo", e)
        self.assertNotIn("bar", e)

    def test_dir(self):
        e = Entity(id="ent1", quantity_factory=self.quantity_factory)
        e.quantities["foo"] = 42
        self.assertIn("foo", dir(e))

    def test_create_quantity(self):
        e = Entity(id="ent1", quantity_factory=self.quantity_factory)
        e.create_quantity("test_quantity")
        self.assertIn("test_quantity", e.quantities)
        self.assertEqual(e.quantities["test_quantity"], "ts_test_quantity")


if __name__ == "__main__":
    unittest.main()
