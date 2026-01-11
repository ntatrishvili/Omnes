import unittest

from app.model.util import InitializingMeta, create_default_quantity
from app.infra.parameter import Parameter
from app.infra.timeseries_object import TimeseriesObject


class TestCreateDefaultQuantity(unittest.TestCase):
    """Tests for the create_default_quantity function."""

    def test_none_returns_none(self):
        self.assertIsNone(create_default_quantity(None))

    def test_scalar_int_returns_parameter(self):
        result = create_default_quantity(42)
        self.assertIsInstance(result, Parameter)
        self.assertEqual(result.value, 42)

    def test_scalar_float_returns_parameter(self):
        result = create_default_quantity(0.95)
        self.assertIsInstance(result, Parameter)
        self.assertEqual(result.value, 0.95)

    def test_string_returns_parameter(self):
        result = create_default_quantity("test")
        self.assertIsInstance(result, Parameter)
        self.assertEqual(result.value, "test")

    def test_bool_is_not_treated_as_scalar(self):
        # Booleans should go to TimeseriesObject path (as iterable falls through)
        result = create_default_quantity(True)
        self.assertIsInstance(result, TimeseriesObject)

    def test_dict_with_value_returns_parameter(self):
        result = create_default_quantity({"value": 100})
        self.assertIsInstance(result, Parameter)
        self.assertEqual(result.value, 100)

    def test_existing_quantity_returned_as_is(self):
        param = Parameter(value=123)
        result = create_default_quantity(param)
        self.assertIs(result, param)


class TestInitializingMeta(unittest.TestCase):
    """Tests for the InitializingMeta metaclass with default_ fields."""

    def test_default_field_converted_to_parameter(self):
        class C(metaclass=InitializingMeta):
            default_capacity = 100.0

            def __init__(self, **kwargs):
                pass

        # Class attribute should be a Parameter (converted at class creation)
        self.assertIsInstance(C.default_capacity, Parameter)
        self.assertEqual(C.default_capacity.value, 100.0)

    def test_instance_inherits_class_default(self):
        class Storage(metaclass=InitializingMeta):
            default_capacity = 100.0
            default_efficiency = 0.9

            def __init__(self, **kwargs):
                pass

        storage = Storage()
        # Instance sees class-level defaults
        self.assertIsInstance(storage.default_capacity, Parameter)
        self.assertEqual(storage.default_capacity.value, 100.0)
        self.assertIsInstance(storage.default_efficiency, Parameter)
        self.assertEqual(storage.default_efficiency.value, 0.9)

    def test_plain_attributes_untouched(self):
        class Device(metaclass=InitializingMeta):
            default_power = 50.0
            other = 7

            def __init__(self, **kwargs):
                pass

        # Non-default_ attributes are left as-is
        self.assertEqual(Device.other, 7)
        Device.other = 42
        self.assertEqual(Device.other, 42)

    def test_excluded_fields_not_converted(self):
        class GridComponent(metaclass=InitializingMeta):
            _quantity_excludes = ["default_phase"]
            default_phase = 3
            default_voltage = 400.0

            def __init__(self, **kwargs):
                pass

        # Excluded field stays as plain value
        self.assertEqual(GridComponent.default_phase, 3)
        self.assertNotIsInstance(GridComponent.default_phase, Parameter)
        # Non-excluded field is converted
        self.assertIsInstance(GridComponent.default_voltage, Parameter)

    def test_exclusion_inherited_from_base(self):
        class Parent(metaclass=InitializingMeta):
            _quantity_excludes = ["default_phase"]
            default_phase = 3

            def __init__(self, **kwargs):
                pass

        class Child(Parent):
            default_phase = "A"  # Override, but still excluded

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        # Child inherits exclusion
        self.assertEqual(Child.default_phase, "A")
        self.assertNotIsInstance(Child.default_phase, Parameter)

    def test_inheritance_of_default_fields(self):
        class Parent(metaclass=InitializingMeta):
            default_value = 1.0

            def __init__(self, **kwargs):
                pass

        class Child(Parent):
            default_extra = 2.0

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        child = Child()
        self.assertIsInstance(child.default_value, Parameter)
        self.assertEqual(child.default_value.value, 1.0)
        self.assertIsInstance(child.default_extra, Parameter)
        self.assertEqual(child.default_extra.value, 2.0)

    def test_multiple_default_fields(self):
        class M(metaclass=InitializingMeta):
            default_name = "bob"
            default_count = 4

            def __init__(self, **kwargs):
                pass

        # String and int defaults become Parameters at class level
        self.assertIsInstance(M.default_name, Parameter)
        self.assertEqual(M.default_name.value, "bob")
        self.assertIsInstance(M.default_count, Parameter)
        self.assertEqual(M.default_count.value, 4)


if __name__ == "__main__":
    unittest.main()
