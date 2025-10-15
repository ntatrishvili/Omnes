"""
Relation and expression system for energy community modeling.

This module provides the core expression system for representing and converting
mathematical relations in energy system optimization models. It supports
arithmetic operations, comparisons, entity references with time offsets,
and specialized expression types for conditional and temporal constraints.
"""

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Union


class Operator(Enum):
    """
    Enumeration of supported mathematical and comparison operators.

    This enum defines all operators that can be used in mathematical expressions
    within the energy system modeling framework. Each operator contains both its
    string representation and category information for proper handling.
    """

    # Comparison operators
    LESS_THAN = ("<", "comparison")
    GREATER_THAN = (">", "comparison")
    LESS_THAN_OR_EQUAL = ("<=", "comparison")
    GREATER_THAN_OR_EQUAL = (">=", "comparison")
    EQUAL = ("==", "comparison")
    NOT_EQUAL = ("!=", "comparison")

    # Arithmetic operators
    ADD = ("+", "arithmetic")
    SUBTRACT = ("-", "arithmetic")
    MULTIPLY = ("*", "arithmetic")
    DIVIDE = ("/", "arithmetic")

    @property
    def symbol(self) -> str:
        """Return the string representation of the operator."""
        return self.value[0]
    
    @property
    def category(self) -> str:
        """Return the category of the operator (comparison or arithmetic)."""
        return self.value[1]
    
    @classmethod
    def from_symbol(cls, symbol: str):
        """Create an operator from its string symbol."""
        for op in cls:
            if op.symbol == symbol:
                return op
        raise ValueError(f"No operator found for symbol: {symbol}")
    
    @classmethod
    def comparison_operators(cls):
        """Return all comparison operators."""
        return [op for op in cls if op.category == "comparison"]
    
    @classmethod
    def arithmetic_operators(cls):
        """Return all arithmetic operators."""
        return [op for op in cls if op.category == "arithmetic"]
    
    @classmethod
    def comparison_strings(cls):
        """Return string representations of comparison operators."""
        return [op.symbol for op in cls.comparison_operators()]
    
    @classmethod
    def arithmetic_strings(cls):
        """Return string representations of arithmetic operators."""
        return [op.symbol for op in cls.arithmetic_operators()]

    def __str__(self) -> str:
        """Return string representation of the operator."""
        return self.symbol


class Expression(ABC):
    """
    Abstract base class for all mathematical expressions.

    This class defines the interface that all expression types must implement
    for parsing, conversion, and entity ID extraction. Expression objects
    form a tree structure representing mathematical operations.
    """

    @abstractmethod
    def __str__(self) -> str:
        """
        Return string representation of the expression.

        Returns
        -------
        str
            Human-readable string representation
        """
        pass

    @abstractmethod
    def get_ids(self) -> List[str]:
        """
        Extract all entity IDs referenced in this expression.

        Returns
        -------
        List[str]
            List of unique entity IDs used in the expression
        """
        pass

    @abstractmethod
    def convert(
        self,
        converter: Any,
        t: int,
        time_set: Optional[int] = None,
        new_freq: Optional[str] = None,
    ) -> Any:
        """
        Convert expression using the provided converter.

        This method implements the visitor pattern, delegating conversion
        logic to the converter object which knows how to handle specific
        target formats (PuLP, pandapower, etc.).

        Parameters
        ----------
        converter : Any
            Converter object that handles format-specific conversion
        t : int
            Current time step
        time_set : int, optional
            Time set for constraint generation
        new_freq : str, optional
            Frequency specification for time conversion

        Returns
        -------
        Any
            Converted representation (format depends on converter)
        """
        pass


class Literal(Expression):
    """
    Expression representing a literal numeric value.

    Literals are terminal nodes in expression trees that contain constant
    numeric values (integers or floats).
    """

    def __init__(self, value: Union[int, float]) -> None:
        """
        Initialize literal with numeric value.

        Parameters
        ----------
        value : Union[int, float]
            The numeric value this literal represents
        """
        self.value = value

    def __str__(self) -> str:
        """Return string representation of the literal value."""
        return str(self.value)

    def get_ids(self) -> List[str]:
        """
        Return empty list as literals contain no entity references.

        Returns
        -------
        List[str]
            Empty list (literals have no entity IDs)
        """
        return []

    def convert(
        self,
        converter: Any,
        t: int,
        time_set: Optional[int] = None,
        new_freq: Optional[str] = None,
    ) -> Any:
        """
        Convert literal using the provided converter.

        Parameters
        ----------
        converter : Any
            Converter object that handles format-specific conversion
        t : int
            Current time step (unused for literals)
        time_set : int, optional
            Time set for constraint generation (unused for literals)
        new_freq : str, optional
            Frequency specification (unused for literals)

        Returns
        -------
        Any
            Converted literal representation
        """
        return converter.convert_literal(self, self.value, t, time_set, new_freq)


class EntityReference(Expression):
    """
    Expression representing a reference to an entity with optional time offset.

    Entity references are used to access variables or properties of energy system
    components at specific time steps. Examples include 'battery1.discharge_power(t)'
    or 'pv1.generation(t-1)'.
    """

    def __init__(self, entity_id: str, time_offset: int = 0) -> None:
        """
        Initialize entity reference.

        Parameters
        ----------
        entity_id : str
            Identifier for the entity (e.g., 'battery1.discharge_power')
        time_offset : int, default=0
            Time offset relative to current time step:
            - 0 for current time (t)
            - -1 for previous time step (t-1)
            - +1 for next time step (t+1)
        """
        if not entity_id or not entity_id.strip():
            raise ValueError("Entity ID cannot be empty")

        self.entity_id = entity_id.strip()
        self.time_offset = time_offset

    def __str__(self) -> str:
        """Return string representation with time offset notation."""
        if self.time_offset == 0:
            return f"{self.entity_id}(t)"
        elif self.time_offset < 0:
            return f"{self.entity_id}(t{self.time_offset})"
        else:
            return f"{self.entity_id}(t+{self.time_offset})"

    def get_ids(self) -> List[str]:
        """
        Return list containing this entity's ID.

        Returns
        -------
        List[str]
            List with single entity ID
        """
        return [self.entity_id]

    def convert(
        self,
        converter: Any,
        t: int,
        time_set: Optional[int] = None,
        new_freq: Optional[str] = None,
    ) -> Any:
        """
        Convert entity reference using the provided converter.

        Parameters
        ----------
        converter : Any
            Converter object that handles format-specific conversion
        t : int
            Current time step
        time_set : int, optional
            Time set for constraint generation
        new_freq : str, optional
            Frequency specification for time conversion

        Returns
        -------
        Any
            Converted entity reference representation
        """
        return converter.convert_entity_reference(
            self, self.entity_id, t, time_set, new_freq
        )


class BinaryExpression(Expression):
    def __init__(self, left: Expression, operator: Operator, right: Expression):
        self.left = left
        self.operator = operator
        self.right = right

    def __str__(self):
        return f"({self.left} {self.operator} {self.right})"

    def get_ids(self) -> list[str]:
        return self.left.get_ids() + self.right.get_ids()

    def convert(self, converter, t: int, time_set: int = None, new_freq: str = None):
        """Convert binary expression using the provided converter"""
        # Recursively convert left and right sides
        left_result = self.left.convert(converter, t, time_set, new_freq)
        right_result = self.right.convert(converter, t, time_set, new_freq)

        # Delegate to converter for the binary operation
        return converter.convert_binary_expression(
            self, left_result, right_result, self.operator, t, time_set, new_freq
        )

    @classmethod
    def parse_binary(cls, expr: str) -> Expression:
        """Parse constraint expressions like 'battery1.discharge_power(t) < 2 * battery1.discharge_power(t-1)'"""
        # Sort comparison operators by length (longest first) to avoid substring conflicts
        comparison_ops = sorted(Operator.comparison_strings(), key=len, reverse=True)
        
        # Find the first matching operator (longest first ensures we get <= instead of <)
        op_str = None
        for op in comparison_ops:
            if op in expr:
                op_str = op
                break

        if not op_str:
            # Early return if no operator found
            return cls._parse_arithmetic_expression(expr)

        # Split on the found operator
        left_str, right_str = expr.split(op_str, 1)

        def _parse_side(expr_part: str):
            return cls._parse_arithmetic_expression(expr_part.strip())

        return BinaryExpression(
            _parse_side(left_str),
            Operator.from_symbol(op_str),
            _parse_side(right_str)
        )

    @classmethod
    def _parse_arithmetic_expression(cls, expr: str) -> Expression:
        """Parse arithmetic expressions with +, -, *, /"""
        # Handle addition and subtraction first (lower precedence, right-to-left)
        for op_str in ["+", "-"]:
            op_pos = cls._find_operator_outside_parentheses(expr, op_str)
            if op_pos != -1:
                left_part = expr[:op_pos].strip()
                right_part = expr[op_pos + 1 :].strip()
                left = cls._parse_arithmetic_expression(left_part)
                right = cls._parse_arithmetic_expression(right_part)
                operator = Operator.from_symbol(op_str)
                return BinaryExpression(left, operator, right)

        # Handle multiplication and division (higher precedence, right-to-left)
        for op_str in ["*", "/"]:
            op_pos = cls._find_operator_outside_parentheses(expr, op_str)
            if op_pos != -1:
                left_part = expr[:op_pos].strip()
                right_part = expr[op_pos + 1 :].strip()
                left = cls._parse_arithmetic_expression(left_part)
                right = cls._parse_term(right_part)
                operator = Operator.from_symbol(op_str)
                return BinaryExpression(left, operator, right)

        # If no arithmetic operators, parse as term
        return cls._parse_term(expr)

    @classmethod
    def _find_operator_outside_parentheses(cls, expr: str, op: str) -> int:
        """Find the last occurrence of an operator that's not inside parentheses"""
        paren_depth = 0
        last_pos = -1

        for i in range(len(expr) - len(op) + 1):
            if expr[i] == "(":
                paren_depth += 1
            elif expr[i] == ")":
                paren_depth -= 1
            elif paren_depth == 0 and expr[i : i + len(op)] == op:
                last_pos = i

        return last_pos

    @classmethod
    def _parse_term(cls, expr: str) -> Expression:
        """Parse individual terms (numbers, entity references)"""
        expr = expr.strip()

        # Check if it's a number first
        try:
            value = float(expr)
            if value.is_integer():
                return Literal(int(value))
            return Literal(value)
        except ValueError:
            pass

        # Check if it's an entity reference with explicit time index
        # Pattern: entity_id.property(t) or entity_id.property(t-1) etc.
        time_pattern = r"^(.+?)\(t([+-]\d+)?\)$"
        match = re.match(time_pattern, expr)
        time_offset = 0
        if match: 
            expr = match.group(1)
            time_offset_str = match.group(2)
            time_offset = 0 if time_offset_str is None else int(time_offset_str)

        # If EntityReference initialization changes, we have to modify only one place in the code
        return EntityReference(expr, time_offset)


class IfThenExpression(Expression):
    def __init__(self, condition_expr, consequence_expr):
        self.condition = condition_expr
        self.consequence = consequence_expr

    def __str__(self):
        return f"(if {self.condition} then {self.consequence})"

    def get_ids(self) -> list[str]:
        return self.condition.get_ids() + self.consequence.get_ids()

    def convert(self, converter, t: int, time_set: int = None, new_freq: str = None):
        """Convert if-then expression using the provided converter"""
        # This would need special handling for conditional constraints
        raise NotImplementedError(
            "If-then expressions require special handling in optimization models"
        )


class TimeConditionExpression(Expression):
    def __init__(self, entity, condition, start_time, end_time):
        self.entity = entity
        self.condition = condition
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return f"({self.entity} {self.condition} from {self.start_time} to {self.end_time})"

    def get_ids(self) -> list[str]:
        # For time conditions, we need to handle the case where condition might be a string
        if hasattr(self.condition, "get_ids"):
            return [self.entity] + self.condition.get_ids()
        else:
            return [self.entity]

    def convert(self, converter, t: int, time_set: int = None, new_freq: str = None):
        """Convert time condition expression using the provided converter"""
        # This would need special time-based constraint handling
        raise NotImplementedError(
            "Time condition expressions require special time-based constraint handling"
        )


class AssignmentExpression(Expression):
    def __init__(self, target, value):
        self.target = target
        self.value = value

    def __str__(self):
        return f"({self.target} = {self.value})"


    def convert(self, converter, t: int, time_set: int = None, new_freq: str = None):
        """Convert assignment expression using the provided converter"""
        # Converter helper: if we are at the leaves, we can specify the type of the object to return, othervise use tha built-in converter of our object
        def _convert(obj, object_as: EntityReference):
            if hasattr(obj, "convert"):
                return obj.convert(converter, t, time_set, new_freq)
        # Literal is OK with str(obj)
            return object_as(str(obj)).convert(converter, t, time_set, new_freq)

        target_result = _convert(self.target, object_as=EntityReference)
        value_result = _convert(self.value, object_as=Literal)
        # Delegate to converter for assignment operation (equality constraint)
        return converter.convert_binary_expression(
                self, target_result, value_result, Operator.EQUAL, t, time_set, new_freq
            )

    def get_ids(self) -> list[str]:
        return self.target.get_ids() + self.value.get_ids()

class Relation:
    def __init__(self, raw_expr: str, name: str):
        self.raw_expr = raw_expr.strip()
        self.name = name
        self.expression = self.parse(self.raw_expr)

    def parse(self, expr: str) -> Expression:
        """Parse the expression string into an Expression tree"""
        # For special expressions, handle them separately
        if expr.startswith("if"):
            return self._parse_if_then(expr)
        if " from " in expr and " to " in expr:
            return self._parse_time_condition(expr)
        if "=" in expr and not any(op in expr for op in Operator.comparison_strings()):
            return self._parse_assignment(expr)

        # For constraint expressions, use BinaryExpression.parse_binary
        return BinaryExpression.parse_binary(expr)

    def _parse_if_then(self, expr: str) -> IfThenExpression:
        _, condition, then_expr = re.split(r"\s*if\s*|\s*then\s*", expr)
        return IfThenExpression(
            self.parse(condition.strip()), self.parse(then_expr.strip())
        )

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

    def get_ids(self) -> list[str]:
        return self.expression.get_ids()

    def convert(
        self, converter, objects: dict, time_set: int = None, new_freq: str = None
    ) -> dict[str, list]:
        """
        Convert this relation using the provided converter for each time step.

        This implements dependency inversion - the relation delegates to the converter,
        which contains all format-specific logic (PuLP, pandapower, etc.).

        Parameters:
        ----------
        converter : object
            The converter that knows how to handle conversion (e.g., PulpConverter)
        objects : dict
            Dictionary mapping entity IDs to their variables/objects
        time_set : int
            The time set to iterate over
        new_freq : str
            Frequency (optional)

        Returns:
        -------
        dict
            Dictionary containing the constraint name and the list of constraints
        """
        return converter.convert_relation(self, objects, time_set, new_freq)

    def __str__(self):
        return f"[{self.name}] {self.expression}"
