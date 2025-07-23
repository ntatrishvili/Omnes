import logging
from enum import Enum, auto
from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import (
    TimeseriesFactory,
    DefaultTimeseriesFactory,
)
from app.model.grid_component.grid_component import GridComponent

logger = logging.getLogger(__name__)


class BusType(Enum):
    PQ = "PQ"
    I = "I"
    Z = "Z"
    SLACK = "SLACK"


class Bus(GridComponent):
    default_nominal_voltage: Optional[float] = None
    default_type: Optional[BusType] = BusType.PQ

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs,
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.quantities.update(
            {
                "voltage": self.ts_factory.create("voltage", **kwargs),
                "nominal_voltage": Parameter(
                    value=kwargs.pop("nominal_voltage", self.default_nominal_voltage)
                ),
                "type": Parameter(value=kwargs.pop("type", self.default_type)),
            }
        )

    def __str__(self):
        """
        String representation of the Bus entity.
        """
        return f"Bus '{self.id}' with nominal voltage={self.nominal_voltage} and type '{self.type.value.value}'"
