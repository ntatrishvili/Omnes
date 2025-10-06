from enum import Enum
from typing import Optional

from app.infra.timeseries_object_factory import (
    TimeseriesFactory,
    DefaultTimeseriesFactory,
)
from app.model.entity import Entity


class Vector(Enum):
    ELECTRICITY = "electricity"
    HEAT = "heat"
    MATERIAL = "material"
    INVALID = "invalid"


class Device(Entity):
    default_vector: Optional[Vector] = Vector.INVALID
    default_contributes_to: Optional[str] = None

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs,
    ):
        super().__init__(id, ts_factory, **kwargs)
        self.bus = kwargs.pop("bus", "bus")
        if self.bus is None:
            raise ValueError(f"No bus specified for device '{self.id}'")
        self.tags = {
            "vector": self.default_vector,
            "contributes_to": self.default_contributes_to,
        }
        self.tags.update(kwargs.pop("tags", {}))

    def update_tags(self, **tags):
        self.tags.update(**tags)
