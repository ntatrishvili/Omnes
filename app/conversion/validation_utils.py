"""
Validation utilities for conversion operations.

This module contains validation functions that are commonly used across
different conversion classes to ensure data integrity and proper error handling.
"""

import warnings
from typing import Optional, Union

_WARNED_MESSAGES = set()


def validate_and_normalize_time_set(
    time_set: Optional[Union[int, range]], default_size: int = 10
) -> range:
    """
    Validate and normalize time_set parameter to a range object.

    Parameters
    ----------
    time_set : Optional[Union[int, range]]
        The time set to validate and normalize. Can be:
        - None: uses default_size
        - int: converted to range(time_set)
        - range: returned as-is
    default_size : int, default=10
        Default size to use if time_set is None

    Returns
    -------
    range
        Normalized time set as a range object

    Raises
    ------
    ValueError
        If time_set is not a valid type or if int time_set is negative

    Examples
    --------
    >>> validate_and_normalize_time_set(5)
    range(0, 5)
    >>> validate_and_normalize_time_set(None, 3)
    range(0, 3)
    """
    if time_set is None:
        return range(default_size)
    elif isinstance(time_set, int):
        if time_set < 0:
            raise ValueError(f"time_set must be non-negative, got {time_set}")
        return range(time_set)
    elif isinstance(time_set, range):
        return time_set
    else:
        raise ValueError(f"time_set must be an int or range, got {type(time_set)}")


def extract_effective_time_properties(model, new_freq, time_set):
    effective_time_set = (
        time_set if time_set is not None else model.number_of_time_steps
    )
    effective_freq = new_freq if new_freq is not None else model.frequency
    return effective_freq, effective_time_set


def validate_entity_exists(entity_id: str, entity_dict: dict) -> None:
    """
    Validate that an entity exists in the provided dictionary.

    Parameters
    ----------
    entity_id : str
        The entity ID to validate
    entity_dict : dict
        Dictionary containing entity mappings

    Raises
    ------
    ValueError
        If entity_id is not found in entity_dict

    Examples
    --------
    >>> entities = {'battery': [1, 2, 3], 'solar': [4, 5, 6]}
    >>> validate_entity_exists('battery', entities)  # No error
    >>> validate_entity_exists('wind', entities)     # Raises ValueError
    """
    if entity_id not in entity_dict:
        raise ValueError(f"Entity ID '{entity_id}' not found in provided objects.")


def handle_time_bounds(
    actual_time: int, pulp_var, variable_name: str = "variable"
) -> int:
    """
    Handle time bounds checking for time-indexed variables with proper warnings.

    Parameters:
    ----------
    actual_time : int
        The requested time index
    pulp_var : Any
        The PuLP variable (usually a list or dict-like object)
    variable_name : str
        Name of the variable for warning messages

    Returns:
    -------
    int
        The adjusted time index within valid bounds
    """
    if isinstance(pulp_var, list):
        if actual_time < 0:
            msg = (
                f"Negative time index {actual_time} for {variable_name} clamped to 0. "
                "This may indicate a logical error in time offset calculations."
            )
            # avoid emitting the same warning repeatedly during tests
            if msg not in _WARNED_MESSAGES:
                warnings.warn(msg, UserWarning)
                _WARNED_MESSAGES.add(msg)
            return 0
        elif actual_time >= len(pulp_var):
            msg = (
                f"Time index {actual_time} for {variable_name} exceeds array bounds "
                f"(size: {len(pulp_var)}). Clamped to last element."
            )
            if msg not in _WARNED_MESSAGES:
                warnings.warn(msg, UserWarning)
                _WARNED_MESSAGES.add(msg)
            return len(pulp_var) - 1
    return actual_time


def validate_linear_programming_constraint(
    left_operand, right_operand, operation: str
) -> None:
    """
    Validate that an operation maintains linear programming constraints.

    Parameters:
    ----------
    left_operand : Any
        Left side of the operation
    right_operand : Any
        Right side of the operation
    operation : str
        Type of operation ("multiplication" or "division")

    Raises:
    ------
    ValueError
        If the operation violates linear programming constraints
    """
    if operation == "multiplication":
        if not isinstance(right_operand, (int, float)) and not isinstance(
            left_operand, (int, float)
        ):
            raise ValueError(
                "Multiplication of two variables is not supported in linear programming. "
                "At least one side must be a constant."
            )
    elif operation == "division":
        if not isinstance(right_operand, (int, float)):
            raise ValueError(
                "Division by variables is not supported in linear programming. "
                "Only division by constants is allowed."
            )
    else:
        raise ValueError(f"Unknown operation type: {operation}")
