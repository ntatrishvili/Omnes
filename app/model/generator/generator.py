from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import (
    DefaultTimeseriesFactory,
    TimeseriesFactory,
)
from app.model.device import Device, Vector


class Generator(Device):
    default_vector = Vector.ELECTRICITY
    default_contributes_to = "electric_power_balance"
    default_peak_power = 0
    default_efficiency = 0

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs
    ):
        super().__init__(id, ts_factory, **kwargs)
        self.quantities.update(
            {
                "peak_power": Parameter(
                    value=kwargs.pop("peak_power", self.default_peak_power)
                ),
                "efficiency": Parameter(
                    value=kwargs.pop("efficiency", self.default_efficiency)
                ),
            }
        )
