from enum import Enum, auto
from typing import Optional

from app.infra.timeseries_object_factory import TimeseriesFactory
from app.model.entity import Entity


class Vector(Enum):
    ELECTRICITY = auto()
    HEAT = auto()
    MATERIAL = auto()
    INVALID = auto()


class Device(Entity):
    default_vector: Optional[Vector] = Vector.INVALID
    default_contributes_to: Optional[str] = None

    def __init__(
        self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
    ):
        super().__init__(id, ts_factory, **kwargs)
        self.bus = kwargs.pop("bus", None)
        if self.bus is None:
            raise ValueError(f"No bus specified for device '{self.id}'")
        self.tags = {
            "vector": kwargs.pop("vector", self.default_vector),
            "contributes_to": kwargs.pop("contributes_to", self.default_contributes_to),
        }
