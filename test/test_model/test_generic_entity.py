import unittest

from app.infra.quantity_factory import QuantityFactory
from app.infra.parameter import Parameter
from app.model.generic_entity import GenericEntity


class DummyQuantityFactory(QuantityFactory):
    def create(self, name, **kwargs):
        return Parameter(value=kwargs.get("value"))


class TestGenericEntity(unittest.TestCase):
    def setUp(self):
        self.quantity_factory = DummyQuantityFactory()

    def test_generic_entity_init_with_kwargs_creates_parameters(self):
        ge = GenericEntity(
            id="g1",
            quantity_factory=self.quantity_factory,
            nominal_power=42,
            label="test",
        )
        # quantities should contain wrapped Parameter objects
        self.assertIn("nominal_power", ge.quantities)
        self.assertIn("label", ge.quantities)
        self.assertIsInstance(ge.nominal_power, Parameter)
        self.assertEqual(ge.nominal_power.value, 42)
        self.assertIsInstance(ge.label, Parameter)
        self.assertEqual(ge.label.value, "test")

    def test_tags_are_forwarded_to_base_entity(self):
        ge = GenericEntity(
            id="g_tags",
            quantity_factory=self.quantity_factory,
            tags={"role": "meta"},
            foo=1,
        )
        self.assertIn("role", ge.tags)
        self.assertEqual(ge.tags["role"], "meta")
        # ensure other kwargs are still converted to quantities
        self.assertIn("foo", ge.quantities)
        self.assertEqual(ge.foo.value, 1)

    def test_quantity_factory_assignment(self):
        ge = GenericEntity(id="g2", quantity_factory=self.quantity_factory, x=5)
        self.assertEqual(ge.quantity_factory, self.quantity_factory)

    def test_str_includes_id_and_parameters(self):
        ge = GenericEntity(id="g_str", quantity_factory=self.quantity_factory, a=7, b=3)
        s = str(ge)
        self.assertIn("Generic entity 'g_str'", s)
        # parameters are converted to strings via their __str__ (numeric values appear)
        self.assertIn("7", s)
        self.assertIn("3", s)


if __name__ == "__main__":
    unittest.main()
