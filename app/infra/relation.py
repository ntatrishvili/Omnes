"""
Relation and expression system for energy community modeling.

This module provides the core expression system for representing and converting
mathematical relations in energy system optimization models. It supports
arithmetic operations, comparisons, entity references with time offsets,
and specialized expression types for conditional and temporal constraints.
"""

import re
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from app.infra.util import TimeSet


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
        time_set: Optional["TimeSet"] = None,
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
        time_set : TimeSet, optional
            TimeSet object containing full time information for constraint generation

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
        time_set: Optional["TimeSet"] = None,
    ) -> Any:
        """
        Convert literal using the provided converter.

        Parameters
        ----------
        converter : Any
            Converter object that handles format-specific conversion
        t : int
            Current time step (unused for literals)
        time_set : TimeSet, optional
            TimeSet object for constraint generation (unused for literals)

        Returns
        -------
        Any
            Converted literal representation
        """
        return converter.convert_literal(self, self.value, t, time_set)


class SelfReference(Expression):
    """
    Expression representing a reference to the containing entity using $ syntax.

    Self references allow relations to refer to properties of their containing entity
    without explicitly naming it. For example, '$.power >= 0' within a battery's
    relations refers to 'battery.power >= 0'.
    """

    def __init__(self, property_name: str, time_offset: int = 0) -> None:
        """
        Initialize self reference.

        Parameters
        ----------
        property_name : str
            The property name (e.g., 'power', 'capacity')
        time_offset : int, default=0
            Time offset relative to current time step:
            - 0 for current time (t)
            - -1 for previous time step (t-1)
            - +1 for next time step (t+1)
        """
        if not property_name or not property_name.strip():
            raise ValueError("Property name cannot be empty")

        self.property_name = property_name.strip()
        self.time_offset = time_offset

    def __str__(self) -> str:
        """Return string representation with $ syntax."""
        if self.time_offset == 0:
            return f"$.{self.property_name}"
        elif self.time_offset < 0:
            return f"$.{self.property_name}(t{self.time_offset})"
        else:
            return f"$.{self.property_name}(t+{self.time_offset})"

    def get_ids(self) -> List[str]:
        """
        Return list with self reference marker.

        Returns
        -------
        List[str]
            List with '$' marker to indicate self reference
        """
        return ["$"]

    def convert(
        self,
        converter: Any,
        t: int,
        time_set: Optional["TimeSet"] = None,
    ) -> Any:
        """
        Convert self reference using the provided converter.

        Parameters
        ----------
        converter : Any
            Converter object that handles format-specific conversion
        t : int
            Current time step
        time_set : TimeSet, optional
            TimeSet object for constraint generation

        Returns
        -------
        Any
            Converted self reference representation
        """
        return converter.convert_self_reference(self, self.property_name, t, time_set)


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
        time_set: Optional["TimeSet"] = None,
    ) -> Any:
        """
        Convert entity reference using the provided converter.

        Parameters
        ----------
        converter : Any
            Converter object that handles format-specific conversion
        t : int
            Current time step
        time_set : TimeSet, optional
            TimeSet object for constraint generation

        Returns
        -------
        Any
            Converted entity reference representation
        """
        return converter.convert_entity_reference(self, self.entity_id, t, time_set)


class BinaryExpression(Expression):
    def __init__(self, left: Expression, operator: Operator, right: Expression):
        self.left = left
        self.operator = operator
        self.right = right

    def __str__(self):
        return f"({self.left} {self.operator} {self.right})"

    def get_ids(self) -> list[str]:
        return self.left.get_ids() + self.right.get_ids()

    def convert(self, converter, t: int, time_set: Optional["TimeSet"] = None):
        """Convert binary expression using the provided converter"""
        # Recursively convert left and right sides
        left_result = self.left.convert(converter, t, time_set)
        right_result = self.right.convert(converter, t, time_set)

        # Delegate to converter for the binary operation
        return converter.convert_binary_expression(
            self, left_result, right_result, self.operator, t, time_set
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
            _parse_side(left_str), Operator.from_symbol(op_str), _parse_side(right_str)
        )

    @classmethod
    def _parse_arithmetic_expression(cls, expr: str) -> Expression:
        """Parse arithmetic expressions with +, -, *, /"""
        expr = expr.strip()

        # Handle parentheses first
        if expr.startswith("(") and expr.endswith(")"):
            # Check if these are balanced outer parentheses
            paren_count = 0
            for i, char in enumerate(expr):
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                    if paren_count == 0 and i < len(expr) - 1:
                        # Not outer parentheses, break and continue with normal parsing
                        break
            else:
                # These are outer parentheses, remove them
                return cls._parse_arithmetic_expression(expr[1:-1])

        # Handle addition and subtraction first (lower precedence, right-to-left)
        for op_str in ["+", "-"]:
            op_pos = cls._find_operator_outside_parentheses(expr, op_str)
            if op_pos != -1:
                left_part = expr[:op_pos].strip()
                right_part = expr[op_pos + 1 :].strip()

                # Special handling for negative numbers at the start
                if op_str == "-" and op_pos == 0:
                    # This is a negative number, not subtraction
                    continue

                # Don't treat it as subtraction if left part is empty or an operator
                if op_str == "-" and (
                    not left_part
                    or left_part.endswith(
                        ("*", "/", "+", "-", "(", "<=", ">=", "<", ">", "==", "!=")
                    )
                ):
                    continue

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
        """Parse individual terms (numbers, entity references, self references)"""
        expr = expr.strip()

        # Check if it's a number first (including negative numbers)
        try:
            value = float(expr)
            if value.is_integer():
                return Literal(int(value))
            return Literal(value)
        except ValueError:
            pass

        # Handle negative numbers that might have been split by operator parsing
        if expr.startswith("-"):
            try:
                # Try to parse the rest as a positive number
                value = float(expr[1:])
                if value.is_integer():
                    return Literal(-int(value))
                return Literal(-value)
            except ValueError:
                pass

        # Check if it's a self reference with $ syntax
        if expr.startswith("$."):
            # Handle $. syntax: $.property or $.property(t-1)
            self_pattern = r"^\$\.(.+?)(\(t([+-]\d+)?\))?$"
            match = re.match(self_pattern, expr)
            if match:
                property_name = match.group(1)
                time_offset_str = match.group(3)
                time_offset = 0 if time_offset_str is None else int(time_offset_str)
                return SelfReference(property_name, time_offset)
            else:
                # Invalid $ syntax
                raise ValueError(f"Invalid self reference syntax: {expr}")

        # Check if it's an entity reference with explicit time index
        # Pattern: entity_id.property(t) or entity_id.property(t-1) etc.
        time_pattern = r"^(.+?)\(t([+-]\d+)?\)$"
        match = re.match(time_pattern, expr)
        time_offset = 0
        if match:
            expr = match.group(1)
            time_offset_str = match.group(2)
            time_offset = 0 if time_offset_str is None else int(time_offset_str)

        # Don't try to create EntityReference with empty string
        if not expr or expr.isspace():
            raise ValueError(f"Cannot parse empty expression term")

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

    def convert(self, converter, t: int, time_set: Optional["TimeSet"] = None):
        """Convert if-then expression using the provided converter"""
        # This would need special handling for conditional constraints
        raise NotImplementedError(
            "If-then expressions require special handling in optimization models"
        )


class TimeConditionExpression(Expression):
    """
    Expression representing a time-based condition for entity constraints.

    This expression allows constraints to be applied only during specific time windows.
    For example, "heater1.p_in enabled from 10:00 to 16:00" means the heater can only
    operate between 10:00 and 16:00.
    """

    def __init__(self, entity, condition, start_time, end_time):
        """
        Initialize time condition expression.

        Parameters
        ----------
        entity : str
            The entity ID (e.g., 'heater1.p_in')
        condition : str
            The condition type ('enabled', 'disabled', etc.)
        start_time : str
            Start time in HH:MM format (e.g., '10:00')
        end_time : str
            End time in HH:MM format (e.g., '16:00')
        """
        self.entity = entity
        self.condition = condition
        self.start_time = start_time
        self.end_time = end_time
        self.__between_times_idx = None
        self.__time_set_id = None

    def __convert_time_to_index(self, time_set: "TimeSet"):
        """
        Convert time window to boolean index based on TimeSet.

        Caches the result based on time_set.hex_id to avoid recomputation.

        Parameters
        ----------
        time_set : TimeSet
            The TimeSet object containing time_points
        """
        if self.__time_set_id == time_set.hex_id:
            return

        self.__between_times_idx = self.__time_to_index(time_set)
        self.__time_set_id = time_set.hex_id

    def __str__(self):
        return f"({self.entity} {self.condition} from {self.start_time} to {self.end_time})"

    def get_ids(self) -> list[str]:
        # For time conditions, we need to handle the case where condition might be a string
        if hasattr(self.condition, "get_ids"):
            return [self.entity] + self.condition.get_ids()
        else:
            return [self.entity]

    def convert(
        self,
        converter,
        t: int,
        time_set: Optional["TimeSet"] = None,
    ):
        """
        Convert time condition expression using the provided converter.

        Parameters
        ----------
        converter : Any
            Converter object that handles format-specific conversion
        t : int
            Current time step
        time_set : TimeSet
            TimeSet object containing full time information including time_points.
            Required for determining which time steps fall within the time window.

        Returns
        -------
        Any
            Converted time condition constraint

        Raises
        ------
        ValueError
            If time_set is None or doesn't have required attributes
        """
        if time_set is None:
            raise ValueError(
                "TimeConditionExpression requires a TimeSet object, got None"
            )
        if not hasattr(time_set, "time_points") or not hasattr(time_set, "hex_id"):
            raise ValueError(
                f"TimeConditionExpression requires a TimeSet object with time_points and hex_id, "
                f"got {type(time_set)}"
            )

        self.__convert_time_to_index(time_set)
        return converter.convert_time_condition_expression(
            self, self.entity, self.condition, self.__between_times_idx, t
        )

    def __time_to_index(self, time_set: "TimeSet"):
        """
        Generate a boolean index indicating which time points are within the time window.

        Parameters
        ----------
        time_set : TimeSet
            The TimeSet object containing time_points

        Returns
        -------
        pandas.DatetimeIndex
            Boolean index where True indicates the time point is within the window
        """
        myarray = np.array([False] * time_set.number_of_time_steps)
        idx = pd.to_datetime(time_set.time_points).indexer_between_time(
            self.start_time, self.end_time
        )
        myarray[idx] = True
        return myarray


class AssignmentExpression(Expression):
    def __init__(self, target, value):
        self.target = target
        self.value = value

    def __str__(self):
        return f"({self.target} = {self.value})"

    def convert(
        self,
        converter,
        t: int,
        time_set: Optional["TimeSet"] = None,
    ):
        """Convert assignment expression using the provided converter"""

        # Converter helper: if we are at the leaves, we can specify the type of the object to return, otherwise use the built-in converter of our object
        def _convert(obj, fallback_type):
            if hasattr(obj, "convert"):
                return obj.convert(converter, t, time_set)
            # For string objects, try to parse them
            obj_str = str(obj).strip()
            if obj_str.startswith("$."):
                # Parse as SelfReference
                self_pattern = r"^\$\.(.+?)(\(t([+-]\d+)?\))?$"
                match = re.match(self_pattern, obj_str)
                if match:
                    property_name = match.group(1)
                    time_offset_str = match.group(3)
                    time_offset = 0 if time_offset_str is None else int(time_offset_str)
                    return SelfReference(property_name, time_offset).convert(
                        converter, t, time_set
                    )
            # Try to parse as number (Literal)
            try:
                value = float(obj_str)
                if value.is_integer():
                    value = int(value)
                return Literal(value).convert(converter, t, time_set)
            except ValueError:
                pass
            # Fall back to EntityReference
            return fallback_type(obj_str).convert(converter, t, time_set)

        target_result = _convert(self.target, EntityReference)
        value_result = _convert(self.value, Literal)
        # Delegate to converter for assignment operation (equality constraint)
        return converter.convert_binary_expression(
            self, target_result, value_result, Operator.EQUAL, t, time_set
        )

    def get_ids(self) -> list[str]:
        target_ids = (
            self.target.get_ids()
            if hasattr(self.target, "get_ids")
            else (
                [str(self.target)]
                if not str(self.target).startswith("$.")
                and not str(self.target).replace(".", "").replace("-", "").isdigit()
                else (["$"] if str(self.target).startswith("$.") else [])
            )
        )
        value_ids = (
            self.value.get_ids()
            if hasattr(self.value, "get_ids")
            else (
                [str(self.value)]
                if not str(self.value).startswith("$.")
                and not str(self.value).replace(".", "").replace("-", "").isdigit()
                else (["$"] if str(self.value).startswith("$.") else [])
            )
        )
        return target_ids + value_ids


class Relation:
    def __init__(self, raw_expr: str, name: str = None):
        self.raw_expr = raw_expr.strip()
        self.name = name if name else f"{uuid.uuid4().hex}"
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
        self,
        converter,
        objects: dict,
        time_set: Optional["TimeSet"] = None,
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
        time_set : TimeSet, optional
            The TimeSet object containing full time information

        Returns:
        -------
        dict
            Dictionary containing the constraint name and the list of constraints
        """
        return converter.convert_relation(self, objects, time_set)

    def __str__(self):
        return f"[{self.name}] {self.expression}"
