from enum import Enum, auto
from typing import Optional

from app.model.grid_component.grid_component import GridComponent
from app.model.timeseries_object_factory import TimeseriesFactory


class BusType(Enum):
    PQ = auto()
    I = auto()
    Z = auto()
    SLACK = auto()


class Bus(GridComponent):
    def __init__(
        self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.quantities = {
            "voltage": self.ts_factory.create("voltage", **kwargs),
            "nominal_voltage": kwargs.get("nominal_voltage", None),
            "phase_count": kwargs.get("phase_count", 1),
            "phase": kwargs.get("phase", "A"),
            "type": kwargs.get("type", BusType.PQ),
        }

    def __str__(self):
        """
        String representation of the Bus entity.
        """
        return f"Bus '{self.id}' with nominal voltage={self['nominal_voltage']} and type '{self['type']}'"
