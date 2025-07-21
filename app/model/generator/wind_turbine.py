from typing import Optional

from app.infra.timeseries_object_factory import (
    TimeseriesFactory,
    DefaultTimeseriesFactory,
)
from app.model.generator.generator import Generator
from app.model.device import Vector


class Wind(Generator):
    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs: object,
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.quantities.update({"p_wind": self.ts_factory.create("p_wind", **kwargs)})

    def __str__(self):
        production_sum = self["p_wind"].sum() if not self["p_wind"].empty else 0
        return f"Wind turbine '{self.id}' with production sum = {production_sum}"
