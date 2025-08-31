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
        """Test that operator enum has correct values"""
        self.assertEqual(Operator.LESS_THAN.value, "<")
        self.assertEqual(Operator.LESS_THAN_OR_EQUAL.value, "<=")
        self.assertEqual(Operator.GREATER_THAN.value, ">")
        self.assertEqual(Operator.GREATER_THAN_OR_EQUAL.value, ">=")
        self.assertEqual(Operator.EQUAL.value, "==")
        self.assertEqual(Operator.NOT_EQUAL.value, "!=")
        self.assertEqual(Operator.ADD.value, "+")
        self.assertEqual(Operator.SUBTRACT.value, "-")
        self.assertEqual(Operator.MULTIPLY.value, "*")
        self.assertEqual(Operator.DIVIDE.value, "/")

    def test_operator_from_string(self):
        """Test creating operators from string"""
        self.assertEqual(Operator.from_string("<"), Operator.LESS_THAN)
        self.assertEqual(Operator.from_string(">="), Operator.GREATER_THAN_OR_EQUAL)
        self.assertEqual(Operator.from_string("*"), Operator.MULTIPLY)

        with self.assertRaises(ValueError):
            Operator.from_string("invalid")


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


class TestEdgeCases(unittest.TestCase):
    def test_parentheses_handling(self):
        """Test that parentheses in time references don't interfere with arithmetic parsing"""
        expr = BinaryExpression.parse_binary(
            "battery.soc(t) >= battery.soc(t-1) + efficiency"
        )

        # Should correctly parse without being confused by (t-1)
        self.assertEqual(expr.operator, Operator.GREATER_THAN_OR_EQUAL)
        self.assertEqual(expr.right.operator, Operator.ADD)
        self.assertEqual(expr.right.left.time_offset, -1)

    def test_simple_variable_names(self):
        """Test parsing simple variable names without dots"""
        expr = BinaryExpression.parse_binary("x + y <= z")

        ids = expr.get_ids()
        self.assertEqual(sorted(ids), ["x", "y", "z"])

        # All should be treated as EntityReference with time_offset=0
        self.assertEqual(expr.left.operator, Operator.ADD)
        self.assertEqual(expr.left.left.time_offset, 0)
        self.assertEqual(expr.left.right.time_offset, 0)

    def test_float_literals(self):
        """Test parsing float literals"""
        expr = BinaryExpression.parse_binary("battery.efficiency >= 0.85")

        self.assertIsInstance(expr.right, Literal)
        self.assertEqual(expr.right.value, 0.85)

    def test_multiple_operators_same_precedence(self):
        """Test left-to-right evaluation for same precedence operators"""
        expr = BinaryExpression.parse_binary("a - b + c")

        # Should parse as: (a - b) + c
        self.assertEqual(expr.operator, Operator.ADD)
        self.assertEqual(expr.left.operator, Operator.SUBTRACT)
        self.assertEqual(expr.right.entity_id, "c")


if __name__ == "__main__":
    unittest.main()
