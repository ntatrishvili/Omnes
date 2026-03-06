from abc import ABC, abstractmethod
from typing import Iterable

from app.infra.parameter import Parameter
from app.infra.quantity import Quantity
from app.infra.timeseries_object import TimeseriesObject


class QuantityFactory(ABC):
    @abstractmethod
    def create(self, quantity_name: str, entity_id: str, **kwargs) -> Quantity: ...


class DefaultQuantityFactory(QuantityFactory):
    @classmethod
    def _handle_input_argument(cls, kwargs):
        raw = kwargs.pop("input")
        if isinstance(raw, Quantity):
            return raw
        if isinstance(raw, dict):
            kwargs.update(raw)
        elif isinstance(raw, (int, float, str)) and not isinstance(raw, bool):
            kwargs["value"] = raw
        elif isinstance(raw, Iterable) and not isinstance(raw, str):
            kwargs["data"] = raw
        else:
            kwargs["value"] = raw
        return None

    @staticmethod
    def _get_named_quantity(kwargs, quantity_name: str):
        return kwargs.get(quantity_name)

    @staticmethod
    def _is_existing_quantity(quantity) -> bool:
        return isinstance(quantity, Quantity)

    @staticmethod
    def _is_numeric_scalar(quantity) -> bool:
        return isinstance(quantity, (int, float)) and not isinstance(quantity, bool)

    @staticmethod
    def _has_timeseries_input(kwargs) -> bool:
        return ("data" in kwargs and isinstance(kwargs["data"], Iterable)) or (
            "input_path" in kwargs
        )

    @staticmethod
    def _has_explicit_value(kwargs) -> bool:
        return "value" in kwargs

    @staticmethod
    def _has_callable_default_type(kwargs) -> bool:
        return "default_type" in kwargs and callable(kwargs["default_type"])

    @staticmethod
    def _build_invalid_arguments_error(quantity_name: str, kwargs) -> ValueError:
        provided_keys = (
            "Received kwargs keys: " + ", ".join(sorted(kwargs.keys()))
            if kwargs
            else "No kwargs provided."
        )
        return ValueError(
            f"Invalid arguments for creating Quantity '{quantity_name}'. "
            f"Expected one of: an existing Quantity under key '{quantity_name}', "
            f"a scalar 'value', iterable 'data', or timeseries arguments "
            f"'input_path' and 'col'. {provided_keys}."
        )

    def create(self, quantity_name: str, entity_id: str, **kwargs) -> Quantity:
        if "input" in kwargs:
            quantity = self._handle_input_argument(kwargs)
            if quantity is not None:
                return quantity

        quantity = self._get_named_quantity(kwargs, quantity_name)
        if self._is_existing_quantity(quantity):
            return quantity

        if self._is_numeric_scalar(quantity):
            return Parameter(value=quantity)

        if self._has_timeseries_input(kwargs):
            return TimeseriesObject(**kwargs, entity_id=entity_id)

        if self._has_explicit_value(kwargs):
            return Parameter(**kwargs)

        if self._has_callable_default_type(kwargs):
            return kwargs.pop("default_type")()

        if quantity is None:
            return Parameter(value=None)

        raise self._build_invalid_arguments_error(quantity_name, kwargs)
