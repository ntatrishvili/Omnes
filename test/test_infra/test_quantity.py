import unittest

from app.infra.quantity import Parameter, Quantity


class TestParameter(unittest.TestCase):
    def test_parameter_value_storage(self):
        param = Parameter(value=42)
        self.assertEqual(param.value, 42)

    def test_parameter_string_representation(self):
        param = Parameter(value=3.14)
        self.assertEqual(str(param), "3.14")

    def test_parameter_get_values(self):
        param = Parameter(value=100)
        self.assertEqual(param.get_values(), 100)

    def test_parameter_empty(self):
        self.assertTrue(Parameter().empty())
        self.assertFalse(Parameter(value=0).empty())
        self.assertFalse(Parameter(value=12.5).empty())

    def test_parameter_equality(self):
        self.assertTrue(Parameter(value=5) == 5)
        self.assertTrue(Parameter(value=5.0) == 5)
        self.assertFalse(Parameter(value=5) == 4)
        self.assertFalse(Parameter(value=5) == "string")

    def test_convert_delegates_to_converter(self):
        class MockConverter:
            def convert_quantity(self, quantity, **kwargs):
                return f"converted_{quantity.value}"

        param = Parameter(value=99)
        result = param.convert(MockConverter())
        self.assertEqual(result, "converted_99")


class TestQuantityABC(unittest.TestCase):
    def test_quantity_is_abstract(self):
        with self.assertRaises(TypeError):
            Quantity()


if __name__ == "__main__":
    unittest.main()
