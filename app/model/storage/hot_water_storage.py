from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import (
    TimeseriesFactory,
    DefaultTimeseriesFactory,
)
from app.model.device import Vector
from app.model.storage.storage import Storage


class HotWaterStorage(Storage):
    default_vector = Vector.HEAT
    default_contributes_to = "heat_balance"
    default_set_temperature: Optional[float] = None
    default_volume: Optional[float] = None

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs,
    ):
        super().__init__(id, ts_factory, **kwargs)
        self.quantities.update(
            {
                "volume": Parameter(value=kwargs.pop("volume", self.default_volume)),
                "set_temperature": Parameter(
                    value=kwargs.pop("set_temperature", self.default_set_temperature)
                ),
            }
        )

    def __str__(self):
        return f"Hot water storage '{self.id}' with volume: {self.quantities['volume']} l and set temperature {self.quantities['set_temperature']} °C"
