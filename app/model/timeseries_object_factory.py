from abc import ABC, abstractmethod

from app.model.timeseries_object import TimeseriesObject


class TimeseriesFactory(ABC):
    @abstractmethod
    def create(self, quantity_name: str, **kwargs) -> TimeseriesObject: ...


class DefaultTimeseriesFactory(TimeseriesFactory):
    def create(self, quantity_name: str, **kwargs) -> TimeseriesObject:
        quantity = kwargs.get(quantity_name, None)
        if quantity is not None and isinstance(quantity, TimeseriesObject):
            return quantity
        return TimeseriesObject(**kwargs)
