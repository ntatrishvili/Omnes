from abc import ABC, abstractmethod
from typing import Iterable

from app.infra.parameter import Parameter
from app.infra.quantity import Quantity
from app.infra.timeseries_object import TimeseriesObject


class QuantityFactory(ABC):
    @abstractmethod
    def create(self, quantity_name: str, entity_id: str, **kwargs) -> Quantity: ...


class DefaultQuantityFactory(QuantityFactory):
    def create(self, quantity_name: str, entity_id: str, **kwargs) -> Quantity:
        # Handle unified 'input' argument
        if "input" in kwargs:
            raw = kwargs.pop("input")
            if isinstance(raw, Quantity):
                return raw
            if isinstance(raw, dict):
                kwargs.update(raw)
            elif isinstance(raw, (int, float, str)) and not isinstance(raw, bool):
                kwargs["value"] = raw
            elif isinstance(raw, Iterable):
                kwargs["data"] = raw
            else:
                kwargs["value"] = raw

        quantity = kwargs.get(quantity_name)
        if isinstance(quantity, Quantity):
            return quantity

        # Scalar heuristic: raw numeric → Parameter
        if isinstance(quantity, (int, float)) and not isinstance(quantity, bool):
            return Parameter(value=quantity)

        # Timeseries heuristics
        if ("data" in kwargs and isinstance(kwargs["data"], Iterable)) or (
            "input_path" in kwargs
        ):
            return TimeseriesObject(**kwargs, entity_id=entity_id)

        # Explicit value → Parameter
        if "value" in kwargs:
            return Parameter(**kwargs)

        # Default type is given, but no other arguments
        if "default_type" in kwargs and callable(kwargs["default_type"]):
            return kwargs.pop("default_type")()

        if quantity is None:
            return Parameter(value=None)

        provided_keys = ", ".join(sorted(kwargs.keys())) or "<none>"
        raise ValueError(
            f"Invalid arguments for creating Quantity '{quantity_name}'. "
            f"Expected one of: an existing Quantity under key '{quantity_name}', "
            f"a scalar 'value', iterable 'data', or timeseries arguments "
            f"'input_path' and 'col'. Received kwargs keys: {provided_keys}."
        )
