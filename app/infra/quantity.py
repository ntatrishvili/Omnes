from abc import ABC, abstractmethod


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
    def value(self, **kwargs):  # NOSONAR
        return None

    @abstractmethod
    def __eq__(self, other): ...

    @abstractmethod
    def empty(self) -> bool: ...
