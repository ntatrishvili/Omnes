from typing import Optional

from app.model.device import Device, Vector
from app.infra.timeseries_object_factory import (
    TimeseriesFactory,
    DefaultTimeseriesFactory,
)


class Load(Device):
    default_vector = Vector.ELECTRICITY
    default_contributes_to = "electric_power_balance"

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs: object,
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.quantities.update({"p_cons": self.ts_factory.create("p_cons", **kwargs)})

    def __str__(self):
        """
        String representation of the Load entity.
        """
        consumption_sum = self.p_cons.sum() if not self.p_cons.empty else 0
        return f"Load '{self.id}' with consumption_sum={consumption_sum}"
