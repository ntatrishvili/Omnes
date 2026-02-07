import logging
from enum import Enum
from typing import Optional

from app.infra.parameter import Parameter
from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.infra.timeseries_object import TimeseriesObject
from app.model.grid_component.grid_component import GridComponent

logger = logging.getLogger(__name__)


class BusType(Enum):
    PQ = "PQ"
    I = "I"
    Z = "Z"
    SLACK = "SLACK"


class Bus(GridComponent):
    _quantity_excludes = ["default_type"]
    default_nominal_voltage: Optional[float] = None
    default_type: Optional[BusType] = BusType.PQ.value

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)
        self.create_quantity(
            "voltage", **kwargs.get("voltage", {}), default_type=TimeseriesObject
        )
        self.create_quantity(
            "nominal_voltage",
            input=kwargs.pop("nominal_voltage", self.default_nominal_voltage),
            default_type=Parameter,
        )
        self.type = BusType(kwargs.pop("type", self.default_type))

    def __str__(self):
        """
        String representation of the Bus entity.
        """
        return f"Bus '{self.id}' with nominal voltage={self.nominal_voltage} and type '{self.type.value}'"
