from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import TimeseriesFactory
from app.model.device import Device
from app.model.load.load import Load


class Converter(Device):
    default_conversion_efficiency: Optional[float] = None

    def __init__(
        self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
    ):
        super().__init__(id, ts_factory, **kwargs)
        self.input_device = kwargs.get("input_device")
        self.output_device = kwargs.get("output_device")
        self.quantities.update(
            {
                "p_in": ts_factory.create("p_in", **kwargs),
                "p_out": ts_factory.create("p_out", **kwargs),
                "conversion_efficiency": Parameter(
                    value=kwargs.get(
                        "conversion_efficiency", self.default_conversion_efficiency
                    )
                ),
            }
        )

    def __str__(self):
        return f"Energy converter '{self.id}' charging device={self['charges']}"


class WaterHeater(Load, Converter):
    default_controllable = True

    def __init__(
        self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
    ):
        super().__init__(id, ts_factory, **kwargs)
        self.output_device = kwargs.pop("charges")
        self.input_device = self.id

    def __str__(self):
        return f"Water Heater '{self.id}' charging device={self['charges']}, ({'controllable' if self['is_controlled'] else 'not controllable'})"
