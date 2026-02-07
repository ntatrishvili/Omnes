from typing import Optional

from app.infra.parameter import Parameter
from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.infra.timeseries_object import TimeseriesObject
from app.model.device import Device


class Transducer(Device):
    default_conversion_efficiency: Optional[float] = None
    default_controllable = True

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        self.input_device = kwargs.pop("input_device", {})
        self.output_device = kwargs.pop("charges")
        if isinstance(self.input_device, Device):
            self.bus = self.input_device.bus
            if "bus" not in kwargs:
                kwargs["bus"] = self.bus
            self.input_device = self.input_device.id
        if isinstance(self.output_device, Device):
            self.output_device = self.output_device.id
        super().__init__(id, quantity_factory, **kwargs)
        self.create_quantity(
            "p_in", **kwargs.get("p_in", {}), default_type=TimeseriesObject
        )
        self.create_quantity(
            "p_out", **kwargs.get("p_out", {}), default_type=TimeseriesObject
        )
        self.create_quantity(
            "controllable",
            input=kwargs.get("controllable", self.default_controllable),
            default_type=TimeseriesObject,
        )
        self.create_quantity(
            "conversion_efficiency",
            input=kwargs.get(
                "conversion_efficiency", self.default_conversion_efficiency
            ),
            default_type=Parameter,
        )

    def __str__(self):
        return f"Energy converter '{self.id}' charging device={self['charges']}"
