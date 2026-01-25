from app.infra.quantity import Quantity
from app.infra.util import cast_like


class Parameter(Quantity):
    """
    Represents a scalar, static quantity used as a parameter in the model.

    This subclass of `Quantity` stores a single numerical value and provides
    methods for exporting it into a format usable by external tools.

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
    def value(self):
        return self._value
