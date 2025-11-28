from abc import ABC, abstractmethod

from app.infra.util import cast_like


class Direction:
    IN = "in"
    OUT = "out"


class Quantity(ABC):
    """
    Abstract base class for representing any abstract quantity (e.g. power flow, financial amount) within an entity.

    Quantities may represent static scalar parameters (Parameter) or dynamic time series (TimeSeriesObject).
    This class defines the interface for how quantities are integrated into
    the optimization model.

    Attributes:
        - Implementations should store any metadata or values passed via **kwargs.
    """

    def __init__(self, **kwargs):
        self.direction = kwargs.pop("direction", None)

    def convert(self, converter, **kwargs):
        """Converts the quantity into a pulp-compatible format (e.g., a time series array or a value-variable)."""
        return converter.convert_quantity(self, **kwargs)

    def set(self, value, **kwargs):
        """Sets the value of the quantity, if applicable."""
        ...

    def set_value(self, value, **kwargs):
        return self.set(value, **kwargs)

    @property
    def value(self, **kwargs):
        return None

    @abstractmethod
    def __eq__(self, other): ...

    @abstractmethod
    def empty(self) -> bool: ...


class Parameter(Quantity):
    """
    Represents a scalar, static quantity used as a parameter in the model.

    This subclass of `Quantity` stores a single numerical value and provides
    methods for exporting it into a format usable by optimization solvers.

    Attributes:
        - value (float or int): The scalar value of the parameter.
    """

    def __init__(self, **kwargs: object):
        super().__init__(**kwargs)
        self._value = kwargs.pop("value", None)

    def __str__(self):
        return f"{self._value}"

    def __eq__(self, other):
        try:
            return float(self._value) == float(other)
        except (TypeError, ValueError):
            return False

    def empty(self) -> bool:
        return self._value is None

    def set(self, value, **kwargs):
        """
        Set the parameter's value after checking type compatibility.

        - If current `_value` is None: accept `value` and store it.
        - Otherwise attempt to convert `value` to the type of `_value`.
        - On failure raise `TypeError`.
        """
        # if current value is not set, accept whatever is provided
        if self._value is None:
            self._value = value
            return

        try:
            converted = cast_like(value, self._value)
        except Exception:
            raise TypeError(
                f"Cannot convert provided value ({value!r}) to type {type(self._value).__name__}"
            )

        self._value = converted

    @property
    def value(self, **kwargs):
        return self._value
