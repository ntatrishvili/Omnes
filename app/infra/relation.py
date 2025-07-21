import re

from pulp import lpSum

from app.infra.util import TimesetBuilder


class Relation:
    """
    Represents a DSL constraint between entity quantities and converts it to a PuLP constraint.
    Supports:
        - Simple expressions:   a < b
        - Conditionals:         if a < b then c < d
        - Time enablement:      device.power enabled from 10:00 to 16:00
        - Duration constraints: device.min_on_duration = 2h
    """

    def __init__(self, expr: str, name: str):
        self.name = name
        self.expr = expr

    @staticmethod
    def _eval_expression(expr: str, context: dict):
        # Evaluate left/right expressions in the context
        return eval(expr, {}, context)

    @staticmethod
    def evaluate_operation(cons_op, left_val, right_val):
        if cons_op == "<":
            return left_val <= right_val
        elif cons_op == "<=":
            return left_val <= right_val
        elif cons_op == ">":
            return left_val >= right_val
        elif cons_op == ">=":
            return left_val >= right_val
        elif cons_op == "==":
            return left_val == right_val
        elif cons_op == "!=":
            return left_val != right_val
        else:
            raise ValueError(f"Unknown operator in consequence: {cons_op}")

    def to_pulp(
        self, context: dict, time_set: int, resolution: str, date: str = "2025-01-01"
    ):
        expr = self.expr.strip().lower()

        if expr.startswith("if"):
            # Pattern: if a < b then c < d
            m = re.match(r"if (.+?) then (.+)", expr)
            if not m:
                raise ValueError(f"Invalid conditional expression: {expr}")
            condition, consequence = m.groups()

            cond_left, cond_op, cond_right = re.split(r"(<=|>=|==|!=|<|>)", condition)
            cons_left, cons_op, cons_right = re.split(r"(<=|>=|==|!=|<|>)", consequence)

            cond_result = self._eval_expression(
                f"{cond_left.strip()} {cond_op} {cond_right.strip()}", context
            )
            if cond_result:
                left_val = self._eval_expression(cons_left.strip(), context)
                right_val = self._eval_expression(cons_right.strip(), context)
                return self.evaluate_operation(cons_op, left_val, right_val)
            else:
                return None  # Condition not true → no constraint

        elif "enabled from" in expr:
            # Pattern: heater2.power enabled from 10:00 to 16:00
            m = re.match(r"(\w+)\.(\w+) enabled from ([\d:]+) to ([\d:]+)", expr)
            if not m:
                raise ValueError(f"Invalid time-enable expression: {expr}")
            entity_id, quantity_name, t_start, t_end = m.groups()
            indices = TimesetBuilder.create(time_set, resolution, date)

            var = context[f"{entity_id}.{quantity_name}"]  # e.g. heater2.p_in
            return [
                var[i] >= 0.01 for i in indices
            ]  # Small threshold to indicate "enabled"

        elif "min_on_duration" in expr:
            # Pattern: heater2.min_on_duration = 2h
            m = re.match(r"(\w+)\.min_on_duration *= *(\d+)h", expr)
            if not m:
                raise ValueError(f"Invalid duration expression: {expr}")
            entity_id, min_duration = m.groups()
            min_duration = int(min_duration)

            on_status = context.get(
                f"{entity_id}.on"
            )  # Expect boolean vars [0/1] for each time step
            if on_status is None:
                raise ValueError(f"Missing .on time series in context for {entity_id}")

            return (
                lpSum(on_status) >= min_duration
            )  # Total ON time steps must reach minimum

        else:
            # Simple expression
            left, op, right = re.split(r"(<=|>=|==|!=|<|>)", self.expr.replace(" ", ""))
            left_val = self._eval_expression(left, context)
            right_val = self._eval_expression(right, context)
            return self.evaluate_operation(op, left_val, right_val)
