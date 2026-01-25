import unittest

from app.infra.relation import (
    AssignmentExpression,
    BinaryExpression,
    EntityReference,
    IfThenExpression,
    Literal,
    Operator,
    Relation,
    TimeConditionExpression,
)


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
        expr = BinaryExpression.parse_binary("battery.soc(t) >= battery.soc(t-1) + 0.1")

        self.assertIsInstance(expr, BinaryExpression)
        self.assertEqual(expr.operator, Operator.GREATER_THAN_OR_EQUAL)

        # Left side: battery.soc(t)
        self.assertEqual(expr.left.entity_id, "battery.soc")
        self.assertEqual(expr.left.time_offset, 0)

        # Right side: battery.soc(t-1) + 0.1
        self.assertIsInstance(expr.right, BinaryExpression)
        self.assertEqual(expr.right.operator, Operator.ADD)
        self.assertEqual(expr.right.left.entity_id, "battery.soc")
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

            def convert_literal(self, literal_obj, value, t, time_set, new_freq):
                # record and return an identifiable value
                self.called = (literal_obj, value, t, time_set, new_freq)
                return ("lit_conv", value)

        conv = C()
        lit = Literal(7)
        res = lit.convert(conv, t=3, time_set=None, new_freq=None)
        self.assertEqual(res, ("lit_conv", 7))
        self.assertIs(conv.called[0], lit)
        self.assertEqual(conv.called[2], 3)

    def test_entityreference_convert_calls_converter(self):
        class C:
            def __init__(self):
                self.called = None

            def convert_entity_reference(
                self, entity_ref_obj, entity_id, t, time_set, new_freq
            ):
                self.called = (entity_ref_obj, entity_id, t)
                return ("ent_conv", entity_id)

        conv = C()
        ref = EntityReference("some.device", -1)
        res = ref.convert(conv, t=5, time_set=10, new_freq="1h")
        self.assertEqual(res, ("ent_conv", "some.device"))
        self.assertIs(conv.called[0], ref)
        self.assertEqual(conv.called[2], 5)

    def test_binary_expression_convert_delegates(self):
        class C:
            def convert_literal(self, literal_obj, value, t, time_set, new_freq):
                return value * 10

            def convert_binary_expression(
                self, expr_obj, left_res, right_res, operator, t, time_set, new_freq
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

            def convert_entity_reference(
                self, entity_ref_obj, entity_id, t, time_set, new_freq
            ):
                return "E", entity_id

            def convert_literal(self, literal_obj, value, t, time_set, new_freq):
                return "L", value

            def convert_binary_expression(
                self, expr_obj, left_res, right_res, operator, t, time_set, new_freq
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

    def test_time_condition_convert_not_implemented(self):
        tc = TimeConditionExpression("heater", "enabled", "08:00", "12:00")
        with self.assertRaises(NotImplementedError):
            tc.convert(None, 0)

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

            def convert_relation(self, relation_obj, objects, time_set, new_freq):
                self.called = (relation_obj, objects, time_set, new_freq)
                return {"converted": True}

        conv = C()
        rel = Relation("a <= b", "deleg")
        objects = {"a": 1, "b": 2}
        res = rel.convert(conv, objects, time_set=10, new_freq="1h")
        self.assertEqual(res, {"converted": True})
        self.assertIs(conv.called[0], rel)
        self.assertEqual(conv.called[1], objects)
        self.assertEqual(conv.called[2], 10)
        self.assertEqual(conv.called[3], "1h")


if __name__ == "__main__":
    unittest.main()
