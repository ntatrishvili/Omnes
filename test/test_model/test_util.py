import unittest

from app.model.util import InitOnSet, InitializingMeta


class TestInitializingMeta(unittest.TestCase):
    def test_default_initialized_at_class_creation(self):
        def dbl(v):
            return 0 if v is None else int(v) * 2

        class C(metaclass=InitializingMeta):
            # default 3 -> initializer should produce 6
            x = InitOnSet(dbl, default=3)

        self.assertEqual(C.x, 6)

    def test_setting_class_attribute_triggers_initializer(self):
        def to_float_or_none(v):
            return None if v is None else float(v)

        class Bus(metaclass=InitializingMeta):
            default_nominal_voltage = InitOnSet(to_float_or_none, default=None)
            # plain attribute should be untouched
            other = 7

        # initial default is None
        self.assertIsNone(Bus.default_nominal_voltage)
        # set string -> converted to float
        Bus.default_nominal_voltage = "11.0"
        self.assertIsInstance(Bus.default_nominal_voltage, float)
        self.assertEqual(Bus.default_nominal_voltage, 11.0)
        # set back to None -> stays None
        Bus.default_nominal_voltage = None
        self.assertIsNone(Bus.default_nominal_voltage)
        # plain attribute is assigned normally
        Bus.other = 42
        self.assertEqual(Bus.other, 42)

    def test_inheritance_of_init_map_and_isolated_assignment(self):
        def i_or_zero(v):
            return int(v) if v is not None else 0

        class Parent(metaclass=InitializingMeta):
            a = InitOnSet(i_or_zero, default=1)

        class Child(Parent):
            pass

        # Both classes start with initialized value
        self.assertEqual(Parent.a, 1)
        self.assertEqual(Child.a, 1)

        # Setting Child.a should apply initializer and not change Parent.a
        Child.a = "5"
        self.assertEqual(Child.a, 5)
        self.assertEqual(Parent.a, 1)

    def test_multiple_init_on_set_attributes(self):
        def s(v):
            return str(v) if v is not None else ""

        def n(v):
            return (int(v) + 1) if v is not None else -1

        class M(metaclass=InitializingMeta):
            name = InitOnSet(s, default="bob")
            count = InitOnSet(n, default=4)

        self.assertEqual(M.name, "bob")
        self.assertEqual(M.count, 5)
        M.name = 123
        M.count = "7"
        self.assertEqual(M.name, "123")
        self.assertEqual(M.count, 8)


if __name__ == "__main__":
    unittest.main()
