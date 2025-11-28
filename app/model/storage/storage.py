from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import (
    DefaultTimeseriesFactory,
    TimeseriesFactory,
)
from app.model.device import Device


class Storage(Device):
    default_capacity: Optional[float] = None
    default_max_charge_rate: Optional[float] = None
    default_max_discharge_rate: Optional[float] = None
    default_charge_efficiency: Optional[float] = 1.0
    default_discharge_efficiency: Optional[float] = 1.0
    default_storage_efficiency: Optional[float] = 1.0

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs
    ):
        super().__init__(id, ts_factory, **kwargs)
        self.create_quantity("p_in", **kwargs)
        self.create_quantity("p_out", **kwargs)
        self.create_quantity("e_stor", **kwargs)
        self.quantities.update(
            {
                "capacity": Parameter(
                    value=kwargs.pop("capacity", self.default_capacity)
                ),
                "max_charge_rate": Parameter(
                    value=kwargs.pop("max_charge_rate", self.default_max_charge_rate)
                ),
                "max_discharge_rate": Parameter(
                    value=kwargs.pop(
                        "max_discharge_rate", self.default_max_discharge_rate
                    )
                ),
                "charge_efficiency": Parameter(
                    value=kwargs.pop(
                        "charge_efficiency", self.default_charge_efficiency
                    )
                ),
                "discharge_efficiency": Parameter(
                    value=kwargs.pop(
                        "discharge_efficiency", self.default_discharge_efficiency
                    )
                ),
                "storage_efficiency": Parameter(
                    value=kwargs.pop(
                        "storage_efficiency", self.default_storage_efficiency
                    )
                ),
            }
        )
