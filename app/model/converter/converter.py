from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import (
    TimeseriesFactory,
    DefaultTimeseriesFactory,
)
from app.model.device import Device
from app.model.load.load import Load


class Converter(Device):
    default_conversion_efficiency: Optional[float] = None
    default_controllable = True

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
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
        super().__init__(id, ts_factory, **kwargs)
        self.controllable = kwargs.pop("controllable", self.default_controllable)
        self.quantities.update(
            {
                "p_in": self.ts_factory.create("p_in", **kwargs),
                "p_out": self.ts_factory.create("p_out", **kwargs),
                "conversion_efficiency": Parameter(
                    value=kwargs.pop(
                        "conversion_efficiency", self.default_conversion_efficiency
                    )
                ),
            }
        )

    def __str__(self):
        return f"Energy converter '{self.id}' charging device={self['charges']}"
