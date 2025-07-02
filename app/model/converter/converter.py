from typing import Optional

from app.infra.timeseries_object_factory import TimeseriesFactory
from app.model.entity import Entity
from app.model.generator.generator import Vector


class Converter(Entity):
    pass


class WaterHeater(Converter):
    default_input_vector = Vector.ELECTRICITY
    default_output_vector = Vector.HEAT
    default_controllable = True

    def __init__(
            self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
    ):
        super().__init__(id, ts_factory, **kwargs)

    def __str__(self):
        return f"Water Heater '{self.id}' (controlled={self['is_controlled']})"
