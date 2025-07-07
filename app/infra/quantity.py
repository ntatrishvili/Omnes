from abc import abstractmethod

from app.infra.util import create_empty_pulp_var


class Quantity:
    """
    Abstract base class for representing any abstract quantity (e.g. power flow, financial amount) within an entity.

    Quantities may represent static scalar parameters (Parameter) or dynamic time series (TimeSeriesObject).
    This class defines the interface for how quantities are integrated into
    the optimization model.

    Attributes:
        - Implementations should store any metadata or values passed via **kwargs.
    """

    def __init__(self, **kwargs): ...

    @abstractmethod
    def to_pulp(self, name: str, freq: str, time_set: int): ...

    @abstractmethod
    def __eq__(self, other): ...

    """Converts the quantity into a pulp-compatible format (e.g., a time series array or a value-variable)."""


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
        self.value = kwargs.pop("value", None)

    def to_pulp(self, name: str, freq: str, time_set: int):
        if self.value is None:
            return create_empty_pulp_var(name, time_set)
        return self.value

    def __str__(self):
        return f"{self.value}"

    def __eq__(self, other):
        try:
            return float(self.value) == float(other)
        except (TypeError, ValueError):
            return False
