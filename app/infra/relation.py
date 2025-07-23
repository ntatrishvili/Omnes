from abc import ABC, abstractmethod


class Expression(ABC):
    @abstractmethod
    def __str__(self):
        pass


class BinaryExpression(Expression):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def __str__(self):
        return f"({self.left} {self.operator} {self.right})"


class IfThenExpression(Expression):
    def __init__(self, condition_expr, consequence_expr):
        self.condition = condition_expr
        self.consequence = consequence_expr

    def __str__(self):
        return f"(if {self.condition} then {self.consequence})"


class TimeConditionExpression(Expression):
    def __init__(self, entity, condition, start_time, end_time):
        self.entity = entity
        self.condition = condition
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return f"({self.entity} {self.condition} from {self.start_time} to {self.end_time})"


class AssignmentExpression(Expression):
    def __init__(self, target, value):
        self.target = target
        self.value = value

    def __str__(self):
        return f"({self.target} = {self.value})"


import re


class Relation:
    def __init__(self, raw_expr: str, name: str):
        self.raw_expr = raw_expr.strip()
        self.name = name
        self.expression = self.parse(self.raw_expr)

    def parse(self, expr: str) -> Expression:
        if expr.startswith("if"):
            return self._parse_if_then(expr)
        if " from " in expr and " to " in expr:
            return self._parse_time_condition(expr)
        if "=" in expr and not any(op in expr for op in ["<", ">", "<=", ">=", "!="]):
            return self._parse_assignment(expr)
        return self._parse_binary(expr)

    def _parse_if_then(self, expr: str) -> IfThenExpression:
        _, condition, then_expr = re.split(r"\s*if\s*|\s*then\s*", expr)
        return IfThenExpression(
            self.parse(condition.strip()), self.parse(then_expr.strip())
        )

    def _parse_binary(self, expr: str) -> BinaryExpression:
        match = re.search(r"(<=|>=|==|!=|<|>)", expr)
        if not match:
            raise ValueError(f"Unsupported binary expression: {expr}")
        operator = match.group(1)
        left, right = expr.split(operator, 1)
        return BinaryExpression(left.strip(), operator, right.strip())

    def _parse_time_condition(self, expr: str) -> TimeConditionExpression:
        match = re.match(
            r"(.+?)\s+(.*?)\s+from\s+(\d{1,2}:\d{2})\s+to\s+(\d{1,2}:\d{2})", expr
        )
        if not match:
            raise ValueError(f"Malformed time condition: {expr}")
        entity, condition, start_time, end_time = match.groups()
        return TimeConditionExpression(
            entity.strip(), condition.strip(), start_time, end_time
        )

    def _parse_assignment(self, expr: str) -> AssignmentExpression:
        left, right = expr.split("=", 1)
        return AssignmentExpression(left.strip(), right.strip())

    def __str__(self):
        return f"[{self.name}] {self.expression}"
