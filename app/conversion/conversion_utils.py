"""
Conversion-specific utility functions for optimization problems.

This module contains utilities specifically for working with conversion functionality
variables, constraints, and expressions.
"""

import warnings

import pulp

from app.infra.relation import Operator

COMPARISON_OPERATORS = set(Operator.comparison_operators())
ARITHMETIC_OPERATORS = set(Operator.arithmetic_operators())


def create_empty_pulp_var(name: str, time_set_size: int) -> list[pulp.LpVariable]:
    """
    Create a list of empty LpVariable with the specified name and time set size.

    Parameters:
    ----------
    name : str
        Base name for the variables
    time_set_size : int
        Number of time steps to create variables for

    Returns:
    -------
    List[pulp.LpVariable]
        List of PuLP variables with non-negative bounds
    """
    if time_set_size < 0:
        raise ValueError(f"time_set_size must be non-negative, got {time_set_size}")
    return [pulp.LpVariable(f"{name}_{t}", lowBound=0) for t in range(time_set_size)]


def is_comparison_operator(operator) -> bool:
    """
    Check if operator is a comparison operator.

    Parameters:
    ----------
    operator : Operator
        The operator to check

    Returns:
    -------
    bool
        True if operator is a comparison operator
    """
    return operator in COMPARISON_OPERATORS


def is_arithmetic_operator(operator) -> bool:
    """
    Check if operator is an arithmetic operator.

    Parameters:
    ----------
    operator : Operator
        The operator to check

    Returns:
    -------
    bool
        True if operator is an arithmetic operator
    """
    return operator in ARITHMETIC_OPERATORS


def handle_comparison_operator(operator, left_result, right_result):
    """
    Handle comparison operators for constraint creation.

    Parameters:
    ----------
    operator : Operator
        The comparison operator
    left_result : Any
        Left operand result
    right_result : Any
        Right operand result

    Returns:
    -------
    PuLP constraint
        The resulting constraint
    """
    if right_result is None:
        raise ValueError(
            f"Right operand in comparison cannot be None, left_result: {left_result}"
        )
    if left_result is None:
        raise ValueError(
            f"Left operand in comparison cannot be None, right_result: {right_result}"
        )
    if operator == Operator.LESS_THAN_OR_EQUAL:
        return left_result <= right_result
    elif operator == Operator.LESS_THAN:
        return left_result <= right_result  # PuLP doesn't support strict inequalities
    elif operator == Operator.GREATER_THAN_OR_EQUAL:
        return left_result >= right_result
    elif operator == Operator.GREATER_THAN:
        return left_result >= right_result  # PuLP doesn't support strict inequalities
    elif operator == Operator.EQUAL:
        return left_result == right_result
    elif operator == Operator.NOT_EQUAL:
        warn_about_not_equal_constraint()
        return left_result != right_result
    return None


def handle_arithmetic_operator(operator, left_result, right_result):
    """
    Handle arithmetic operators for expression creation.

    Parameters:
    ----------
    operator : Operator
        The arithmetic operator
    left_result : Any
        Left operand result
    right_result : Any
        Right operand result

    Returns:
    -------
    PuLP expression
        The resulting arithmetic expression

    Raises:
    ------
    ValueError
        If the operation violates linear programming constraints
    """
    if operator == Operator.ADD:
        return left_result + right_result
    elif operator == Operator.SUBTRACT:
        return left_result - right_result
    elif operator == Operator.MULTIPLY:
        validate_linear_multiplication(left_result, right_result)
        return left_result * right_result
    elif operator == Operator.DIVIDE:
        validate_linear_division(right_result)
        return left_result * (1.0 / right_result)


def warn_about_not_equal_constraint() -> None:
    """Warn user about complexity of not-equal constraints."""
    warnings.warn(
        "Not equal constraints (!=) are supported in PuLP via Mixed Integer Programming, "
        "but require binary variables and increase complexity significantly. "
        "Consider using strict inequalities (< or >) if possible.",
        UserWarning,
    )


def validate_linear_multiplication(left_result, right_result) -> None:
    """
    Validate that multiplication maintains linearity.

    Parameters:
    ----------
    left_result : Any
        Left operand
    right_result : Any
        Right operand

    Raises:
    ------
    ValueError
        If both operands are variables (non-linear)
    """
    if not isinstance(right_result, (int, float)) and not isinstance(
        left_result, (int, float)
    ):
        raise ValueError(
            "Multiplication of two variables is not supported in linear programming. "
            "At least one side must be a constant."
        )


def validate_linear_division(right_result) -> None:
    """
    Validate that division maintains linearity.

    Parameters:
    ----------
    right_result : Any
        The divisor

    Raises:
    ------
    ValueError
        If divisor is a variable (non-linear)
    """
    if not isinstance(right_result, (int, float)):
        raise ValueError(
            "Division by variables is not supported in linear programming. "
            "Only division by constants is allowed."
        )
