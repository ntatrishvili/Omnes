"""
Validation utilities for conversion operations.

This module contains validation functions that are commonly used across
different conversion classes to ensure data integrity and proper error handling.
"""

import warnings
from typing import Optional, Union

_WARNED_MESSAGES = set()


def validate_and_normalize_time_range(time_range, default_size: int = 10) -> range:
    """
    Validate and normalize time_range parameter to a range object.

    Parameters
    ----------
    time_range : Optional[Union[int, range, TimeSet]]
        The time range to validate and normalize. Can be:
        - None: uses default_size
        - int: converted to range(time_range)
        - range: returned as-is
        - TimeSet: uses number_of_time_steps
    default_size : int, default=10
        Default size to use if time_range is None

    Returns
    -------
    range
        Normalized time range as a range object

    Raises
    ------
    ValueError
        If time_range is not a valid type or if int time_range is negative

    Examples
    --------
    >>> validate_and_normalize_time_range(5)
    range(0, 5)
    >>> validate_and_normalize_time_range(None, 3)
    range(0, 3)
    """
    if time_range is None:
        return range(default_size)
    elif isinstance(time_range, int):
        if time_range < 0:
            raise ValueError(f"time_range must be non-negative, got {time_range}")
        return range(time_range)
    elif isinstance(time_range, range):
        return time_range
    elif hasattr(time_range, "number_of_time_steps"):
        # Handle TimeSet objects
        return range(time_range.number_of_time_steps)
    else:
        raise ValueError(
            f"time_range must be an int, range, or TimeSet, got {type(time_range)}"
        )


def extract_effective_time_properties(model, time_set):
    """
    Extract effective time properties from model and override parameters.

    Creates a TimeSet object with the effective frequency and time range,
    using provided overrides or falling back to model defaults.

    Parameters
    ----------
    model : Model
        The model containing default time properties
    time_set : TimeSet or int, optional
        Override time set or number of time steps. If None, uses model defaults

    Returns
    -------
    TimeSet
        A TimeSet object with the effective time configuration
    """
    from app.infra.util import TimesetBuilder

    if time_set is None:
        return model.time_set
    # Use TimesetBuilder to create the TimeSet with proper pandas date_range
    # TimesetBuilder handles the case where time_start is None by defaulting to 1970-01-01
    return TimesetBuilder.create(
        time_start=time_set.start or model.time_set.start,
        time_end=time_set.end or model.time_set.end,
        resolution=time_set.freq or model.frequencyt,
        number_of_time_steps=time_set.number_of_time_steps
        or model.number_of_time_steps,
        tz=time_set.tz or model.time_set.tz,
    )


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
