import unittest

import pytest

from app.infra.relation import (
    Relation,
    BinaryExpression,
    IfThenExpression,
    TimeConditionExpression,
    AssignmentExpression,
)


# Test BinaryExpression parsing
def test_binary_expression_parsing():
    relation = Relation(
        "battery1.max_discharge_rate < 2 * pv1.peak_power", "BatteryDischargeRate"
    )
    expr = relation.expression
    assert isinstance(expr, BinaryExpression)
    assert expr.left == "battery1.max_discharge_rate"
    assert expr.operator == "<"
    assert expr.right == "2 * pv1.peak_power"
    assert str(expr) == "(battery1.max_discharge_rate < 2 * pv1.peak_power)"


# Test IfThenExpression parsing
def test_if_then_expression_parsing():
    relation = Relation(
        "if battery1.capacity < 6 then battery1.max_discharge_rate < 3",
        "BatteryCapacity",
    )
    expr = relation.expression
    assert isinstance(expr, IfThenExpression)
    assert isinstance(expr.condition, BinaryExpression)
    assert isinstance(expr.consequence, BinaryExpression)
    assert expr.condition.left == "battery1.capacity"
    assert expr.consequence.right == "3"
    assert (
        str(expr)
        == "(if (battery1.capacity < 6) then (battery1.max_discharge_rate < 3))"
    )


# Test TimeConditionExpression parsing
def test_time_condition_expression_parsing():
    relation = Relation("heater2.power enabled from 10:00 to 16:00", "HeaterEnabled")
    expr = relation.expression
    assert isinstance(expr, TimeConditionExpression)
    assert expr.entity == "heater2.power"
    assert expr.condition == "enabled"
    assert expr.start_time == "10:00"
    assert expr.end_time == "16:00"
    assert str(expr) == "(heater2.power enabled from 10:00 to 16:00)"


# Test AssignmentExpression parsing
def test_assignment_expression_parsing():
    relation = Relation("heater2.min_on_duration = 2h", "HeaterMinOnDuration")
    expr = relation.expression
    assert isinstance(expr, AssignmentExpression)
    assert expr.target == "heater2.min_on_duration"
    assert expr.value == "2h"
    assert str(expr) == "(heater2.min_on_duration = 2h)"


# Test invalid expression raises error
def test_invalid_expression_raises():
    with pytest.raises(ValueError):
        Relation("this is not valid", "InvalidTest")


if __name__ == "__main__":
    unittest.main()
