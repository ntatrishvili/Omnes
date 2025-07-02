from enum import Enum, auto
from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import TimeseriesFactory
from app.model.entity import Entity


class Vector(Enum):
    ELECTRICITY = auto()
    HEAT = auto()
    MATERIAL = auto()
    INVALID = auto()


class Generator(Entity):
    default_vector = Vector.INVALID
    default_contributes_to = ""
    default_peak_power = 0
    default_efficiency = 0

    def __init__(self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs):
        super().__init__(id, ts_factory, **kwargs)
        self.bus = kwargs.pop("bus")
        self.quantities.update({"peak_power": Parameter(value=kwargs.pop("peak_power", self.default_peak_power)),
                                "efficiency": Parameter(value=kwargs.pop("efficiency", self.default_efficiency))})
        self.tags = {"vector": kwargs.pop("vector", self.default_vector),
                     "contributes_to": kwargs.pop("contributes_to", self.default_contributes_to)}
