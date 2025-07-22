import unittest

import pytest
from pulp import LpVariable, LpBinary, LpContinuous

from app.infra.relation import Relation


class TestRelation(unittest.TestCase):
    def setUp(self):
        self.dummy_context = {
            "a": 3,
            "b": 5,
            "c": 7,
            "d": 9,
            "device.power": [
                LpVariable(f"power_{i}", cat=LpContinuous) for i in range(6)
            ],
            "heater2.on": [LpVariable(f"on_{i}", cat=LpBinary) for i in range(4)],
        }

    def test_simple_expression(self):
        rel = Relation("a < b", "simple_test")
        result = rel.convert(self.dummy_context, time_set=4, resolution="1h")
        assert result is True

    def test_conditional_true(self):
        rel = Relation("if a < b then c < d", "cond_test_true")
        result = rel.convert(self.dummy_context, time_set=4, resolution="1h")
        assert result is True

    def test_conditional_false(self):
        rel = Relation("if a > b then c < d", "cond_test_false")
        result = rel.convert(self.dummy_context, time_set=4, resolution="1h")
        assert result is None  # condition not satisfied

    def test_time_enablement(self):
        rel = Relation("device.power enabled from 10:00 to 15:00", "enable_test")
        result = rel.convert(
            self.dummy_context, time_set=6, resolution="1h", date="2025-01-01"
        )
        assert isinstance(result, list)
        assert all(constraint.name().startswith("power_") for constraint in result)
        assert len(result) > 0

    def test_min_on_duration(self):
        rel = Relation("heater2.min_on_duration = 2h", "duration_test")
        result = rel.convert(
            self.dummy_context, time_set=4, resolution="1h", date="2025-01-01"
        )
        assert result.name.startswith("on_0") or "Sum" in result.name()

    def test_invalid_expression(self):
        with pytest.raises(ValueError):
            rel = Relation("if a < b then then c < d", "invalid_expr")
            rel.convert({}, 4, "1h")


if __name__ == "__main__":
    unittest.main()
