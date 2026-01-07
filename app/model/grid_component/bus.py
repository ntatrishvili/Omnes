import logging
from enum import Enum
from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import (
    DefaultTimeseriesFactory,
    TimeseriesFactory,
)
from app.model.grid_component.grid_component import GridComponent
from app.model.util import InitOnSet

logger = logging.getLogger(__name__)


class BusType(Enum):
    PQ = "PQ"
    I = "I"
    Z = "Z"
    SLACK = "SLACK"


class Bus(GridComponent):
    default_nominal_voltage: Optional[float] = InitOnSet(
        lambda v: (
            None if v is None else DefaultTimeseriesFactory().create("voltage", **v)
        ),
        default=None,
    )
    default_type: Optional[BusType] = InitOnSet(
        lambda v: BusType(v) if v is not None else BusType.PQ, default=BusType.PQ
    )

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
                "type": Parameter(value=BusType(kwargs.pop("type", self.default_type))),
            }
        )

    def __str__(self):
        """
        String representation of the Bus entity.
        """
        return f"Bus '{self.id}' with nominal voltage={self.nominal_voltage} and type '{self.type.value.value}'"
