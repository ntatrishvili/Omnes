from abc import ABC, abstractmethod
from typing import Iterable

from app.infra.parameter import Parameter
from app.infra.quantity import Quantity
from app.infra.timeseries_object import TimeseriesObject


class QuantityFactory(ABC):
    @abstractmethod
    def create(self, quantity_name: str, **kwargs) -> Quantity: ...


class DefaultQuantityFactory(QuantityFactory):
    def create(self, quantity_name: str, **kwargs) -> Quantity:
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
            "input_path" in kwargs and "col" in kwargs
        ):
            return TimeseriesObject(**kwargs)

        # Explicit value → Parameter
        if "value" in kwargs:
            return Parameter(**kwargs)

        if quantity is None:
            return Parameter(value=None)

        raise ValueError("Invalid arguments for creating Quantity")
