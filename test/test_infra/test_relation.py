import unittest.mock

import pandas as pd

from app.infra.relation import (
    AssignmentExpression,
    BinaryExpression,
    EntityReference,
    IfThenExpression,
    Literal,
    Operator,
    Relation,
    SelfReference,
    TimeConditionExpression,
)

# Alias for Mock to make tests cleaner
Mock = unittest.mock.Mock


class TestOperator(unittest.TestCase):
    def test_operator_enum_values(self):
        """Test that operator enum has correct symbol and category values"""
        self.assertEqual(Operator.LESS_THAN.symbol, "<")
        self.assertEqual(Operator.LESS_THAN.category, "comparison")
        self.assertEqual(Operator.LESS_THAN_OR_EQUAL.symbol, "<=")
        self.assertEqual(Operator.LESS_THAN_OR_EQUAL.category, "comparison")
        self.assertEqual(Operator.GREATER_THAN.symbol, ">")
        self.assertEqual(Operator.GREATER_THAN.category, "comparison")
        self.assertEqual(Operator.GREATER_THAN_OR_EQUAL.symbol, ">=")
        self.assertEqual(Operator.GREATER_THAN_OR_EQUAL.category, "comparison")
        self.assertEqual(Operator.EQUAL.symbol, "==")
        self.assertEqual(Operator.EQUAL.category, "comparison")
        self.assertEqual(Operator.NOT_EQUAL.symbol, "!=")
        self.assertEqual(Operator.NOT_EQUAL.category, "comparison")
        self.assertEqual(Operator.ADD.symbol, "+")
        self.assertEqual(Operator.ADD.category, "arithmetic")
        self.assertEqual(Operator.SUBTRACT.symbol, "-")
        self.assertEqual(Operator.SUBTRACT.category, "arithmetic")
        self.assertEqual(Operator.MULTIPLY.symbol, "*")
        self.assertEqual(Operator.MULTIPLY.category, "arithmetic")
        self.assertEqual(Operator.DIVIDE.symbol, "/")
        self.assertEqual(Operator.DIVIDE.category, "arithmetic")

    def test_operator_from_string(self):
        """Test creating operators from string using from_symbol method"""
        self.assertEqual(Operator.from_symbol("<"), Operator.LESS_THAN)
        self.assertEqual(Operator.from_symbol(">="), Operator.GREATER_THAN_OR_EQUAL)
        self.assertEqual(Operator.from_symbol("*"), Operator.MULTIPLY)

        with self.assertRaises(ValueError):
            Operator.from_symbol("invalid")

    def test_operator_category_methods(self):
        """Test category-based operator grouping methods"""
        comparison_ops = Operator.comparison_operators()
        arithmetic_ops = Operator.arithmetic_operators()

        self.assertEqual(len(comparison_ops), 6)
        self.assertEqual(len(arithmetic_ops), 4)

        # Test that all comparison operators have the right category
        for op in comparison_ops:
            self.assertEqual(op.category, "comparison")

        # Test that all arithmetic operators have the right category
        for op in arithmetic_ops:
            self.assertEqual(op.category, "arithmetic")

        # Test string methods
        comparison_strings = Operator.comparison_strings()
        arithmetic_strings = Operator.arithmetic_strings()

        self.assertIn("<", comparison_strings)
        self.assertIn("<=", comparison_strings)
        self.assertIn("+", arithmetic_strings)
        self.assertIn("*", arithmetic_strings)


class TestLiteral(unittest.TestCase):
    def test_literal_integer(self):
        """Test literal with integer value"""
        lit = Literal(42)
        self.assertEqual(lit.value, 42)
        self.assertEqual(str(lit), "42")
        self.assertEqual(lit.get_ids(), [])

    def test_literal_float(self):
        """Test literal with float value"""
        lit = Literal(3.14)
        self.assertEqual(lit.value, 3.14)
        self.assertEqual(str(lit), "3.14")
        self.assertEqual(lit.get_ids(), [])


class TestSelfReference(unittest.TestCase):
    def test_self_reference_current_time(self):
        """Test self reference with current time (t)"""
        ref = SelfReference("power", 0)
        self.assertEqual(ref.property_name, "power")
        self.assertEqual(ref.time_offset, 0)
        self.assertEqual(str(ref), "$.power")
        self.assertEqual(ref.get_ids(), ["$"])

    def test_self_reference_past_time(self):
        """Test self reference with past time (t-1)"""
        ref = SelfReference("power", -1)
        self.assertEqual(ref.property_name, "power")
        self.assertEqual(ref.time_offset, -1)
        self.assertEqual(str(ref), "$.power(t-1)")
        self.assertEqual(ref.get_ids(), ["$"])

    def test_self_reference_future_time(self):
        """Test self reference with future time (t+1)"""
        ref = SelfReference("power", 2)
        self.assertEqual(ref.property_name, "power")
        self.assertEqual(ref.time_offset, 2)
        self.assertEqual(str(ref), "$.power(t+2)")
        self.assertEqual(ref.get_ids(), ["$"])

    def test_self_reference_empty_property_raises(self):
        """Test that empty property name raises ValueError"""
        with self.assertRaises(ValueError) as context:
            SelfReference("")
        self.assertIn("Property name cannot be empty", str(context.exception))

    def test_self_reference_whitespace_property_raises(self):
        """Test that whitespace-only property name raises ValueError"""
        with self.assertRaises(ValueError) as context:
            SelfReference("   ")
        self.assertIn("Property name cannot be empty", str(context.exception))

    def test_self_reference_strips_whitespace(self):
        """Test that property name whitespace is stripped"""
        ref = SelfReference("  power  ")
        self.assertEqual(ref.property_name, "power")

    def test_self_reference_convert_calls_converter(self):
        """Test that convert delegates to converter"""

        class MockConverter:
            def __init__(self):
                self.called = None

            def convert_self_reference(self, self_ref, property_name, t, time_set):
                self.called = (self_ref, property_name, t, time_set)
                return ("self_conv", property_name)

        conv = MockConverter()
        ref = SelfReference("capacity", -2)
        res = ref.convert(conv, t=7, time_set=None)
        self.assertEqual(res, ("self_conv", "capacity"))
        self.assertIs(conv.called[0], ref)
        self.assertEqual(conv.called[1], "capacity")
        self.assertEqual(conv.called[2], 7)


class TestEntityReference(unittest.TestCase):
    def test_entity_reference_current_time(self):
        """Test entity reference with current time (t)"""
        ref = EntityReference("battery1.power", 0)
        self.assertEqual(ref.entity_id, "battery1.power")
        self.assertEqual(ref.time_offset, 0)
        self.assertEqual(str(ref), "battery1.power(t)")
        self.assertEqual(ref.get_ids(), ["battery1.power"])

    def test_entity_reference_past_time(self):
        """Test entity reference with past time (t-1)"""
        ref = EntityReference("battery1.power", -1)
        self.assertEqual(ref.entity_id, "battery1.power")
        self.assertEqual(ref.time_offset, -1)
        self.assertEqual(str(ref), "battery1.power(t-1)")
        self.assertEqual(ref.get_ids(), ["battery1.power"])

    def test_entity_reference_future_time(self):
        """Test entity reference with future time (t+1)"""
        ref = EntityReference("battery1.power", 2)
        self.assertEqual(ref.entity_id, "battery1.power")
        self.assertEqual(ref.time_offset, 2)
        self.assertEqual(str(ref), "battery1.power(t+2)")
        self.assertEqual(ref.get_ids(), ["battery1.power"])


class TestBinaryExpressionParsing(unittest.TestCase):
    def test_parse_simple_comparison(self):
        """Test parsing simple comparison expressions"""
        expr = BinaryExpression.parse_binary("battery1.power <= 100")

        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.LESS_THAN_OR_EQUAL)

        # Check left side
        self.assertIsInstance(expr.left, EntityReference)
        self.assertEqual(expr.left.entity_id, "battery1.power")
        self.assertEqual(expr.left.time_offset, 0)

        # Check right side
        self.assertIsInstance(expr.right, Literal)
        self.assertEqual(expr.right.value, 100)

    def test_parse_arithmetic_expression(self):
        """Test parsing arithmetic expressions"""
        expr = BinaryExpression.parse_binary("battery1.charging_power < 2 * pv1.power")

        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.LESS_THAN)

        # Left side should be entity reference
        self.assertIsInstance(expr.left, EntityReference)
        self.assertEqual(expr.left.entity_id, "battery1.charging_power")

        # Right side should be multiplication
        self.assertIsInstance(expr.right, BinaryExpression)
        self.assertEqual(expr.right.operator, Operator.MULTIPLY)
        self.assertIsInstance(expr.right.left, Literal)
        self.assertEqual(expr.right.left.value, 2)
        self.assertIsInstance(expr.right.right, EntityReference)
        self.assertEqual(expr.right.right.entity_id, "pv1.power")

    def test_parse_complex_time_references(self):
        """Test parsing expressions with explicit time references"""
        expr = BinaryExpression.parse_binary(
            "battery1.discharge_power(t) < 2 * battery1.discharge_power(t-1)"
        )

        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.LESS_THAN)

        # Left side: battery1.discharge_power(t)
        self.assertIsInstance(expr.left, EntityReference)
        self.assertEqual(expr.left.entity_id, "battery1.discharge_power")
        self.assertEqual(expr.left.time_offset, 0)

        # Right side: 2 * battery1.discharge_power(t-1)
        self.assertIsInstance(expr.right, BinaryExpression)
        self.assertEqual(expr.right.operator, Operator.MULTIPLY)
        self.assertEqual(expr.right.left.value, 2)
        self.assertEqual(expr.right.right.entity_id, "battery1.discharge_power")
        self.assertEqual(expr.right.right.time_offset, -1)

    def test_parse_addition_subtraction(self):
        """Test parsing addition and subtraction with correct precedence"""
        expr = BinaryExpression.parse_binary(
            "battery.state_of_charge(t) >= battery.state_of_charge(t-1) + 0.1"
        )

        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.GREATER_THAN_OR_EQUAL)

        # Left side: battery.state_of_charge(t)
        self.assertEqual(expr.left.entity_id, "battery.state_of_charge")
        self.assertEqual(expr.left.time_offset, 0)

        # Right side: battery.state_of_charge(t-1) + 0.1
        self.assertIsInstance(expr.right, BinaryExpression)
        self.assertEqual(expr.right.operator, Operator.ADD)
        self.assertEqual(expr.right.left.entity_id, "battery.state_of_charge")
        self.assertEqual(expr.right.left.time_offset, -1)
        self.assertEqual(expr.right.right.value, 0.1)

    def test_operator_precedence(self):
        """Test that multiplication has higher precedence than addition"""
        expr = BinaryExpression.parse_binary("x + 2 * y")

        # Should parse as: x + (2 * y)
        # Our parser finds the rightmost + first, then handles * in the right side
        self.assertEqual(expr.operator, Operator.ADD)
        self.assertEqual(expr.left.entity_id, "x")

        # Right side should be multiplication
        self.assertEqual(expr.right.operator, Operator.MULTIPLY)
        self.assertEqual(expr.right.left.value, 2)
        self.assertEqual(expr.right.right.entity_id, "y")

    def test_get_ids(self):
        """Test that get_ids returns all entity IDs in the expression"""
        expr = BinaryExpression.parse_binary(
            "battery1.power + grid.import_power <= total_demand"
        )
        ids = expr.get_ids()

        expected_ids = ["battery1.power", "grid.import_power", "total_demand"]
        self.assertEqual(sorted(ids), sorted(expected_ids))


class TestRelation(unittest.TestCase):
    def test_relation_constraint_expression(self):
        """Test that Relation correctly delegates to BinaryExpression.parse_binary"""
        relation = Relation("battery1.power <= 100", "test_constraint")

        self.assertEqual(relation.name, "test_constraint")
        self.assertEqual(relation.raw_expr, "battery1.power <= 100")
        self.assertIsInstance(relation.expression, BinaryExpression)

        # Test that it produces the same result as direct parsing
        direct_parse = BinaryExpression.parse_binary("battery1.power <= 100")
        self.assertEqual(str(relation.expression), str(direct_parse))

    def test_relation_get_ids(self):
        """Test that Relation.get_ids works correctly"""
        relation = Relation("battery1.power + pv1.power <= grid.capacity", "test")
        ids = relation.get_ids()

        expected_ids = ["battery1.power", "pv1.power", "grid.capacity"]
        self.assertEqual(sorted(ids), sorted(expected_ids))

    def test_relation_str(self):
        """Test string representation of Relation"""
        relation = Relation("x <= y", "test_constraint")
        expected = "[test_constraint] (x(t) <= y(t))"
        self.assertEqual(str(relation), expected)

    def test_if_then_expression(self):
        """Test if-then expression parsing"""
        relation = Relation(
            "if battery1.capacity < 6 then battery1.max_discharge_rate < 3",
            "conditional",
        )

        self.assertIsInstance(relation.expression, IfThenExpression)
        self.assertIsInstance(relation.expression.condition, BinaryExpression)
        self.assertIsInstance(relation.expression.consequence, BinaryExpression)

    def test_time_condition_expression(self):
        """Test time condition expression parsing"""
        relation = Relation(
            "heater2.power enabled from 10:00 to 16:00", "time_condition"
        )

        self.assertIsInstance(relation.expression, TimeConditionExpression)
        self.assertEqual(relation.expression.entity, "heater2.power")
        self.assertEqual(relation.expression.condition, "enabled")
        self.assertEqual(relation.expression.start_time, "10:00")
        self.assertEqual(relation.expression.end_time, "16:00")

    def test_assignment_expression(self):
        """Test assignment expression parsing"""
        relation = Relation("heater2.min_on_duration = 2h", "assignment")

        self.assertIsInstance(relation.expression, AssignmentExpression)
        self.assertEqual(relation.expression.target, "heater2.min_on_duration")
        self.assertEqual(relation.expression.value, "2h")


class TestAdditionalRelationCases(unittest.TestCase):
    def test_entity_reference_invalid_id_raises(self):
        with self.assertRaises(ValueError):
            EntityReference("  ", 0)

    def test_literal_convert_calls_converter(self):
        class C:
            def __init__(self):
                self.called = None

            def convert_literal(self, literal_obj, value, t, time_set):
                # record and return an identifiable value
                self.called = (literal_obj, value, t, time_set)
                return ("lit_conv", value)

        conv = C()
        lit = Literal(7)
        res = lit.convert(conv, t=3, time_set=None)
        self.assertEqual(res, ("lit_conv", 7))
        self.assertIs(conv.called[0], lit)
        self.assertEqual(conv.called[2], 3)

    def test_entityreference_convert_calls_converter(self):
        class C:
            def __init__(self):
                self.called = None

            def convert_entity_reference(self, entity_ref_obj, entity_id, t, time_set):
                self.called = (entity_ref_obj, entity_id, t)
                return ("ent_conv", entity_id)

        conv = C()
        ref = EntityReference("some.device", -1)
        res = ref.convert(conv, t=5, time_set=10)
        self.assertEqual(res, ("ent_conv", "some.device"))
        self.assertIs(conv.called[0], ref)
        self.assertEqual(conv.called[2], 5)

    def test_binary_expression_convert_delegates(self):
        class C:
            def convert_literal(self, literal_obj, value, t, time_set):
                return value * 10

            def convert_binary_expression(
                self, expr_obj, left_res, right_res, operator, t, time_set
            ):
                return (expr_obj, left_res, right_res, operator)

        conv = C()
        be = BinaryExpression(Literal(2), Operator.MULTIPLY, Literal(3))
        res = be.convert(conv, t=0)
        # left 2 -> 20, right 3 -> 30, operator MULTIPLY
        self.assertEqual(res[1], 20)
        self.assertEqual(res[2], 30)
        self.assertEqual(res[3], Operator.MULTIPLY)

    def test_assignment_expression_convert_uses_entity_and_literal(self):
        class C:
            def __init__(self):
                self.called = None

            def convert_entity_reference(self, entity_ref_obj, entity_id, t, time_set):
                return "E", entity_id

            def convert_literal(self, literal_obj, value, t, time_set):
                return "L", value

            def convert_binary_expression(
                self, expr_obj, left_res, right_res, operator, t, time_set
            ):
                # return a compact tuple for assertions
                return left_res, right_res, operator

        conv = C()
        assign = AssignmentExpression("sensor.temp", "25")
        res = assign.convert(conv, t=1)
        # left should be handled via convert_entity_reference and right via convert_literal
        # Note: "25" gets parsed as integer 25 by the _convert method
        self.assertEqual(res[0], ("E", "sensor.temp"))
        self.assertEqual(res[1], ("L", 25))
        self.assertEqual(res[2], Operator.EQUAL)

    def test_time_condition_convert_requires_time_set(self):
        """TimeConditionExpression.convert requires a TimeSet object"""
        tc = TimeConditionExpression("heater", "enabled", "08:00", "12:00")
        with self.assertRaises(ValueError) as context:
            tc.convert(None, 0)
        self.assertIn(
            "TimeConditionExpression requires a TimeSet object", str(context.exception)
        )

    def test_parse_numeric_literals_int_and_float(self):
        expr_int = BinaryExpression.parse_binary("2.0")
        self.assertIsInstance(expr_int, Literal)
        self.assertEqual(expr_int.value, 2)

        expr_float = BinaryExpression.parse_binary("2.5")
        self.assertIsInstance(expr_float, Literal)
        self.assertEqual(expr_float.value, 2.5)

    def test_get_ids_with_duplicates(self):
        # left is 'a', right is (a + b) -> get_ids returns ['a','a','b']
        inner = BinaryExpression(
            EntityReference("a"), Operator.ADD, EntityReference("b")
        )
        top = BinaryExpression(EntityReference("a"), Operator.ADD, inner)
        ids = top.get_ids()
        self.assertEqual(ids, ["a", "a", "b"])

    def test_time_condition_malformed_raises(self):
        # malformed time string should raise ValueError in parsing
        with self.assertRaises(ValueError):
            Relation("heater.power enabled from 100 to 16:00")

    def test_assignment_get_ids_with_expressions(self):
        # AssignmentExpression with expression objects should return their ids
        assign = AssignmentExpression(EntityReference("a"), Literal(5))
        ids = assign.get_ids()
        self.assertEqual(ids, ["a"])

    def test_if_then_convert_not_implemented(self):
        rel = Relation("if a < b then c < d")
        with self.assertRaises(NotImplementedError):
            rel.expression.convert(None, 0)

    def test_operator_str(self):
        # cover Operator.__str__
        self.assertEqual(str(Operator.ADD), "+")
        self.assertEqual(str(Operator.LESS_THAN_OR_EQUAL), "<=")

    def test_find_operator_outside_parentheses(self):
        # operator only inside parentheses -> should return -1
        self.assertEqual(
            BinaryExpression._find_operator_outside_parentheses("(a+b)", "+"), -1
        )
        # multiple operators, ensure last outside-paren operator is found
        expr = "a+(b+c)+d"
        idx = BinaryExpression._find_operator_outside_parentheses(expr, "+")
        # '+' at index 7 is the last '+' outside parentheses in this compact string
        self.assertEqual(idx, 7)

    def test_parse_with_parentheses_and_multiplication(self):
        # top-level multiplication outside parentheses should be detected
        expr = BinaryExpression.parse_binary("a*(b + c)")
        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.MULTIPLY)
        # left is 'a'
        self.assertIsInstance(expr.left, EntityReference)
        self.assertEqual(expr.left.entity_id, "a")
        # right remains as the literal parenthesized term
        self.assertIsInstance(expr.right, EntityReference)
        self.assertEqual(expr.right.entity_id, "(b + c)")

    def test_parse_explicit_positive_time_offset(self):
        expr = BinaryExpression.parse_binary("pv1.power(t+3)")
        self.assertIsInstance(expr, EntityReference)
        self.assertEqual(expr.entity_id, "pv1.power")
        self.assertEqual(expr.time_offset, 3)

    def test_relation_convert_delegates(self):
        class C:
            def __init__(self):
                self.called = None

            def convert_relation(self, relation_obj, objects, time_set):
                self.called = (relation_obj, objects, time_set)
                return {"converted": True}

        conv = C()
        rel = Relation("a <= b", "deleg")
        objects = {"a": 1, "b": 2}
        res = rel.convert(conv, objects, time_set=10)
        self.assertEqual(res, {"converted": True})
        self.assertIs(conv.called[0], rel)
        self.assertEqual(conv.called[1], objects)
        self.assertEqual(conv.called[2], 10)


class TestSelfReferenceParsing(unittest.TestCase):
    """Tests for parsing self references in expressions"""

    def test_parse_self_reference_no_offset(self):
        """Test parsing $.property without time offset"""
        expr = BinaryExpression._parse_term("$.power")
        self.assertIsInstance(expr, SelfReference)
        self.assertEqual(expr.property_name, "power")
        self.assertEqual(expr.time_offset, 0)

    def test_parse_self_reference_with_t(self):
        """Test parsing $.property(t)"""
        expr = BinaryExpression._parse_term("$.power(t)")
        self.assertIsInstance(expr, SelfReference)
        self.assertEqual(expr.property_name, "power")
        self.assertEqual(expr.time_offset, 0)

    def test_parse_self_reference_negative_offset(self):
        """Test parsing $.property(t-1)"""
        expr = BinaryExpression._parse_term("$.capacity(t-1)")
        self.assertIsInstance(expr, SelfReference)
        self.assertEqual(expr.property_name, "capacity")
        self.assertEqual(expr.time_offset, -1)

    def test_parse_self_reference_positive_offset(self):
        """Test parsing $.property(t+2)"""
        expr = BinaryExpression._parse_term("$.soc(t+2)")
        self.assertIsInstance(expr, SelfReference)
        self.assertEqual(expr.property_name, "soc")
        self.assertEqual(expr.time_offset, 2)

    def test_parse_invalid_self_reference_raises(self):
        """Test that invalid self reference syntax raises ValueError"""
        with self.assertRaises(ValueError) as context:
            BinaryExpression._parse_term("$.")
        self.assertIn("Invalid self reference syntax", str(context.exception))

    def test_parse_self_reference_in_binary_expression(self):
        """Test parsing self reference within a binary expression"""
        expr = BinaryExpression.parse_binary("$.power(t) >= 0")
        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.GREATER_THAN_OR_EQUAL)
        self.assertIsInstance(expr.left, SelfReference)
        self.assertEqual(expr.left.property_name, "power")

    def test_parse_self_reference_with_arithmetic(self):
        """Test parsing self reference in arithmetic expression"""
        expr = BinaryExpression.parse_binary("$.soc(t) - $.soc(t-1) <= $.max_rate")
        self.assertIsInstance(expr, BinaryExpression)
        # Should have self references on both sides
        ids = expr.get_ids()
        self.assertEqual(ids, ["$", "$", "$"])


class TestTimeConditionExpressionExtended(unittest.TestCase):
    """Extended tests for TimeConditionExpression"""

    def test_time_condition_str(self):
        """Test string representation of TimeConditionExpression"""
        tc = TimeConditionExpression("heater.power", "enabled", "09:00", "17:00")
        self.assertEqual(str(tc), "(heater.power enabled from 09:00 to 17:00)")

    def test_time_condition_get_ids_with_expression_condition(self):
        """Test get_ids when condition is an expression with get_ids method"""
        mock_condition = Mock()
        mock_condition.get_ids.return_value = ["sensor.value"]
        tc = TimeConditionExpression("heater.power", mock_condition, "09:00", "17:00")
        ids = tc.get_ids()
        self.assertEqual(ids, ["heater.power", "sensor.value"])

    def test_time_condition_convert_missing_time_points(self):
        """Test that convert raises when time_set lacks time_points"""
        tc = TimeConditionExpression("heater.power", "enabled", "09:00", "17:00")
        mock_time_set = Mock(spec=["hex_id"])  # Missing time_points
        with self.assertRaises(ValueError) as context:
            tc.convert(Mock(), 0, time_set=mock_time_set)
        self.assertIn("time_points and hex_id", str(context.exception))

    def test_time_condition_convert_missing_hex_id(self):
        """Test that convert raises when time_set lacks hex_id"""
        tc = TimeConditionExpression("heater.power", "enabled", "09:00", "17:00")
        mock_time_set = Mock(spec=["time_points"])  # Missing hex_id
        with self.assertRaises(ValueError) as context:
            tc.convert(Mock(), 0, time_set=mock_time_set)
        self.assertIn("time_points and hex_id", str(context.exception))

    def test_time_condition_convert_success(self):
        """Test successful conversion with valid time_set"""
        tc = TimeConditionExpression("heater.power", "enabled", "10:00", "16:00")

        mock_converter = Mock()
        mock_converter.convert_time_condition_expression.return_value = "constraint"

        # Create a proper mock time_set
        mock_time_set = Mock()
        mock_time_set.hex_id = "abc123"
        mock_time_set.number_of_time_steps = 24
        mock_time_set.time_points = pd.date_range("2024-01-01", periods=24, freq="h")

        result = tc.convert(mock_converter, t=5, time_set=mock_time_set)
        self.assertEqual(result, "constraint")
        mock_converter.convert_time_condition_expression.assert_called_once()

    def test_time_condition_caches_time_index(self):
        """Test that time index is cached based on hex_id"""
        tc = TimeConditionExpression("heater.power", "enabled", "10:00", "14:00")

        mock_converter = Mock()
        mock_converter.convert_time_condition_expression.return_value = "constraint"

        mock_time_set = Mock()
        mock_time_set.hex_id = "same_id"
        mock_time_set.number_of_time_steps = 24
        mock_time_set.time_points = pd.date_range("2024-01-01", periods=24, freq="h")

        # Call convert twice with same hex_id
        tc.convert(mock_converter, t=0, time_set=mock_time_set)
        tc.convert(mock_converter, t=1, time_set=mock_time_set)

        # Should be called twice
        self.assertEqual(mock_converter.convert_time_condition_expression.call_count, 2)

    def test_time_condition_recomputes_on_different_hex_id(self):
        """Test that time index is recomputed when hex_id changes"""
        tc = TimeConditionExpression("heater.power", "enabled", "10:00", "14:00")

        mock_converter = Mock()
        mock_converter.convert_time_condition_expression.return_value = "constraint"

        mock_time_set1 = Mock()
        mock_time_set1.hex_id = "id1"
        mock_time_set1.number_of_time_steps = 24
        mock_time_set1.time_points = pd.date_range("2024-01-01", periods=24, freq="h")

        mock_time_set2 = Mock()
        mock_time_set2.hex_id = "id2"
        mock_time_set2.number_of_time_steps = 24
        mock_time_set2.time_points = pd.date_range("2024-01-02", periods=24, freq="h")

        # Call with different time_sets
        tc.convert(mock_converter, t=0, time_set=mock_time_set1)
        tc.convert(mock_converter, t=0, time_set=mock_time_set2)

        self.assertEqual(mock_converter.convert_time_condition_expression.call_count, 2)


class TestAssignmentExpressionExtended(unittest.TestCase):
    """Extended tests for AssignmentExpression"""

    def test_assignment_str(self):
        """Test string representation"""
        assign = AssignmentExpression("device.power", "100")
        self.assertEqual(str(assign), "(device.power = 100)")

    def test_assignment_get_ids_numeric_target(self):
        """Test get_ids with numeric string target (edge case)"""
        assign = AssignmentExpression("123", "value")
        ids = assign.get_ids()
        # Numeric target should result in empty list for target
        self.assertIn("value", ids)

    def test_assignment_get_ids_numeric_value(self):
        """Test get_ids with numeric string value"""
        assign = AssignmentExpression("device.power", "42")
        ids = assign.get_ids()
        self.assertEqual(ids, ["device.power"])  # Numeric value excluded

    def test_assignment_get_ids_self_reference_target(self):
        """Test get_ids with self reference as target"""
        assign = AssignmentExpression("$.power", "100")
        ids = assign.get_ids()
        self.assertIn("$", ids)

    def test_assignment_get_ids_self_reference_value(self):
        """Test get_ids with self reference as value"""
        assign = AssignmentExpression("device.power", "$.max_power")
        ids = assign.get_ids()
        self.assertIn("device.power", ids)
        self.assertIn("$", ids)

    def test_assignment_get_ids_both_self_references(self):
        """Test get_ids when both target and value are self references"""
        assign = AssignmentExpression("$.output", "$.input")
        ids = assign.get_ids()
        self.assertEqual(ids, ["$", "$"])

    def test_assignment_convert_with_self_reference_target(self):
        """Test conversion with self reference target string"""

        class MockConverter:
            def convert_self_reference(self, ref, prop, t, ts):
                return ("self", prop)

            def convert_literal(self, lit, val, t, ts):
                return ("lit", val)

            def convert_binary_expression(self, expr, left, right, op, t, ts):
                return (left, right, op)

        conv = MockConverter()
        assign = AssignmentExpression("$.power", "50")
        result = assign.convert(conv, t=0)
        self.assertEqual(result[0], ("self", "power"))
        self.assertEqual(result[1], ("lit", 50))
        self.assertEqual(result[2], Operator.EQUAL)

    def test_assignment_convert_with_self_reference_value(self):
        """Test conversion with self reference value string"""

        class MockConverter:
            def convert_entity_reference(self, ref, eid, t, ts):
                return ("entity", eid)

            def convert_self_reference(self, ref, prop, t, ts):
                return ("self", prop)

            def convert_binary_expression(self, expr, left, right, op, t, ts):
                return (left, right, op)

        conv = MockConverter()
        assign = AssignmentExpression("device.power", "$.max_power")
        result = assign.convert(conv, t=0)
        self.assertEqual(result[0], ("entity", "device.power"))
        self.assertEqual(result[1], ("self", "max_power"))

    def test_assignment_convert_with_self_reference_time_offset(self):
        """Test conversion with self reference with time offset"""

        class MockConverter:
            def __init__(self):
                self.self_ref_calls = []

            def convert_self_reference(self, ref, prop, t, ts):
                self.self_ref_calls.append((ref.time_offset, prop))
                return ("self", prop, ref.time_offset)

            def convert_literal(self, lit, val, t, ts):
                return ("lit", val)

            def convert_binary_expression(self, expr, left, right, op, t, ts):
                return (left, right, op)

        conv = MockConverter()
        assign = AssignmentExpression("$.soc(t-1)", "100")
        assign.convert(conv, t=0)
        # Check that time offset was parsed
        self.assertEqual(conv.self_ref_calls[0][0], -1)

    def test_assignment_convert_with_float_value(self):
        """Test conversion with float string value"""

        class MockConverter:
            def convert_entity_reference(self, ref, eid, t, ts):
                return ("entity", eid)

            def convert_literal(self, lit, val, t, ts):
                return ("lit", val)

            def convert_binary_expression(self, expr, left, right, op, t, ts):
                return (left, right, op)

        conv = MockConverter()
        assign = AssignmentExpression("device.temp", "3.14")
        result = assign.convert(conv, t=0)
        self.assertEqual(result[1], ("lit", 3.14))


class TestBinaryExpressionParsingExtended(unittest.TestCase):
    """Extended tests for BinaryExpression parsing"""

    def test_parse_division(self):
        """Test parsing division expressions"""
        expr = BinaryExpression.parse_binary("power / 2 <= 50")
        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.LESS_THAN_OR_EQUAL)
        self.assertIsInstance(expr.left, BinaryExpression)
        self.assertEqual(expr.left.operator, Operator.DIVIDE)

    def test_parse_subtraction_expression(self):
        """Test parsing subtraction expressions"""
        expr = BinaryExpression.parse_binary("a - b <= 10")
        self.assertIsInstance(expr, BinaryExpression)
        self.assertIsInstance(expr.left, BinaryExpression)
        self.assertEqual(expr.left.operator, Operator.SUBTRACT)

    def test_parse_negative_number_at_start(self):
        """Test parsing negative number at start of expression"""
        expr = BinaryExpression._parse_arithmetic_expression("-5")
        self.assertIsInstance(expr, Literal)
        self.assertEqual(expr.value, -5)

    def test_parse_negative_float(self):
        """Test parsing negative float"""
        expr = BinaryExpression._parse_arithmetic_expression("-3.14")
        self.assertIsInstance(expr, Literal)
        self.assertEqual(expr.value, -3.14)

    def test_parse_negative_integer_via_term(self):
        """Test parsing negative integer through _parse_term"""
        expr = BinaryExpression._parse_term("-42")
        self.assertIsInstance(expr, Literal)
        self.assertEqual(expr.value, -42)

    def test_parse_empty_term_raises(self):
        """Test that empty term raises ValueError"""
        with self.assertRaises(ValueError) as context:
            BinaryExpression._parse_term("")
        self.assertIn("Cannot parse empty expression", str(context.exception))

    def test_parse_whitespace_term_raises(self):
        """Test that whitespace-only term raises ValueError"""
        with self.assertRaises(ValueError) as context:
            BinaryExpression._parse_term("   ")
        self.assertIn("Cannot parse empty expression", str(context.exception))

    def test_parse_subtraction_after_multiply(self):
        """Test that subtraction after multiply is handled correctly"""
        expr = BinaryExpression._parse_arithmetic_expression("a * -3")
        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.MULTIPLY)

    def test_parse_subtraction_after_divide(self):
        """Test that subtraction after divide is handled correctly"""
        expr = BinaryExpression._parse_arithmetic_expression("a / -2")
        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.DIVIDE)

    def test_parse_subtraction_after_add(self):
        """Test that subtraction after add is handled correctly"""
        expr = BinaryExpression._parse_arithmetic_expression("a + -5")
        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.ADD)

    def test_parse_subtraction_after_subtract(self):
        """Test that subtraction with parentheses is handled correctly"""
        # Use parentheses to clarify intent for nested subtractions
        expr = BinaryExpression._parse_arithmetic_expression("10 - (5 - 3)")
        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.SUBTRACT)

    def test_parse_subtraction_after_open_paren(self):
        """Test that subtraction after open paren is handled correctly"""
        expr = BinaryExpression._parse_arithmetic_expression("(-5)")
        self.assertIsInstance(expr, Literal)
        self.assertEqual(expr.value, -5)

    def test_parse_all_comparison_operators(self):
        """Test parsing all comparison operators"""
        operators = {
            "<": Operator.LESS_THAN,
            ">": Operator.GREATER_THAN,
            "<=": Operator.LESS_THAN_OR_EQUAL,
            ">=": Operator.GREATER_THAN_OR_EQUAL,
            "==": Operator.EQUAL,
            "!=": Operator.NOT_EQUAL,
        }
        for symbol, expected_op in operators.items():
            expr = BinaryExpression.parse_binary(f"x {symbol} y")
            self.assertEqual(expr.operator, expected_op)

    def test_parse_nested_parentheses(self):
        """Test parsing nested parentheses"""
        expr = BinaryExpression._parse_arithmetic_expression("((a + b))")
        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.ADD)

    def test_parse_entity_without_time_index(self):
        """Test parsing entity reference without explicit time index"""
        expr = BinaryExpression._parse_term("battery.power")
        self.assertIsInstance(expr, EntityReference)
        self.assertEqual(expr.entity_id, "battery.power")
        self.assertEqual(expr.time_offset, 0)


class TestRelationExtended(unittest.TestCase):
    """Extended tests for Relation class"""

    def test_relation_auto_generated_name(self):
        """Test that relation generates UUID name when not provided"""
        rel = Relation("a <= b")
        # UUID hex is 32 characters
        self.assertEqual(len(rel.name), 32)
        self.assertTrue(rel.name.isalnum())

    def test_relation_strips_whitespace(self):
        """Test that relation strips whitespace from raw expression"""
        rel = Relation("  a <= b  ", "test")
        self.assertEqual(rel.raw_expr, "a <= b")

    def test_relation_if_then_get_ids(self):
        """Test get_ids for if-then expression"""
        rel = Relation("if a < 5 then b > 3")
        ids = rel.get_ids()
        self.assertIn("a", ids)
        self.assertIn("b", ids)

    def test_relation_time_condition_disabled(self):
        """Test time condition with disabled keyword"""
        rel = Relation("pump.power disabled from 22:00 to 06:00")
        self.assertIsInstance(rel.expression, TimeConditionExpression)
        self.assertEqual(rel.expression.condition, "disabled")

    def test_relation_assignment_with_self_reference(self):
        """Test assignment expression with self reference"""
        rel = Relation("$.power = 100")
        self.assertIsInstance(rel.expression, AssignmentExpression)
        self.assertEqual(rel.expression.target, "$.power")

    def test_relation_complex_arithmetic_with_parentheses(self):
        """Test complex arithmetic with parentheses"""
        rel = Relation("(a + b) * c <= 100")
        self.assertIsInstance(rel.expression, BinaryExpression)
        self.assertEqual(rel.expression.operator, Operator.LESS_THAN_OR_EQUAL)

    def test_relation_negative_bound(self):
        """Test relation with negative bound"""
        rel = Relation("power >= -100")
        self.assertIsInstance(rel.expression, BinaryExpression)
        self.assertIsInstance(rel.expression.right, Literal)
        self.assertEqual(rel.expression.right.value, -100)

    def test_relation_float_bound(self):
        """Test relation with float bound"""
        rel = Relation("efficiency <= 0.95")
        self.assertIsInstance(rel.expression.right, Literal)
        self.assertEqual(rel.expression.right.value, 0.95)


class TestIfThenExpression(unittest.TestCase):
    """Tests for IfThenExpression"""

    def test_if_then_str(self):
        """Test string representation"""
        condition = BinaryExpression(
            EntityReference("a"), Operator.LESS_THAN, Literal(5)
        )
        consequence = BinaryExpression(
            EntityReference("b"), Operator.GREATER_THAN, Literal(3)
        )
        expr = IfThenExpression(condition, consequence)
        self.assertIn("if", str(expr))
        self.assertIn("then", str(expr))

    def test_if_then_get_ids(self):
        """Test get_ids returns IDs from both condition and consequence"""
        condition = EntityReference("sensor.temp")
        consequence = EntityReference("heater.power")
        expr = IfThenExpression(condition, consequence)
        ids = expr.get_ids()
        self.assertIn("sensor.temp", ids)
        self.assertIn("heater.power", ids)

    def test_if_then_convert_raises_not_implemented(self):
        """Test that convert raises NotImplementedError"""
        condition = Literal(1)
        consequence = Literal(2)
        expr = IfThenExpression(condition, consequence)
        with self.assertRaises(NotImplementedError):
            expr.convert(Mock(), t=0)


if __name__ == "__main__":
    unittest.main()
