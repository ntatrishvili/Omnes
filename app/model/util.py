from app.infra.quantity import Quantity
from app.infra.parameter import Parameter
from app.infra.timeseries_object import TimeseriesObject


def create_default_quantity(value):
    """
    Convert a raw value into a Parameter or TimeseriesObject.
    Used for initializing default_ class members at class creation time.
    """
    if value is None:
        return None
    if isinstance(value, Quantity):
        return value
    if isinstance(value, (int, float, str)) and not isinstance(value, bool):
        return Parameter(value=value)
    if isinstance(value, dict):
        if "value" in value:
            return Parameter(**value)
        return TimeseriesObject(**value)
    # Iterable -> TimeseriesObject
    return TimeseriesObject(data=value)


class InitializingMeta(type):
    """
    Metaclass that converts default_ class members into Parameter/TimeseriesObject
    at class creation time. Does not interfere with instance initialization.

    Fields listed in _quantity_excludes are left as plain values.
    """

    def __new__(mcs, name, bases, namespace):
        # Collect exclusion list from class and all ancestors
        excluded = set(namespace.get("_quantity_excludes", []))
        for base in bases:
            for ancestor in getattr(base, "__mro__", []):
                excluded.update(getattr(ancestor, "_quantity_excludes", []))

        # Convert default_ fields to Quantity at class level
        for key, value in list(namespace.items()):
            if key.startswith("default_") and key not in excluded:
                if not isinstance(value, Quantity):
                    namespace[key] = create_default_quantity(value)

        return super().__new__(mcs, name, bases, namespace)
