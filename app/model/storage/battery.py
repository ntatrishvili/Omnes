from typing import Optional

from .storage import Storage
from ..device import Vector
from ...infra.timeseries_object_factory import (
    TimeseriesFactory,
    DefaultTimeseriesFactory,
)


class Battery(Storage):
    default_vector = Vector.ELECTRICITY
    default_contributes_to = "electric_power_balance"

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs,
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.quantities.update(
            {
                "p_bess_in": self.ts_factory.create("p_bess_in", **kwargs),
                "p_bess_out": self.ts_factory.create("p_bess_out", **kwargs),
                "e_bess_stor": self.ts_factory.create("e_bess_stor", **kwargs),
            }
        )


    def __str__(self):
        """
        String representation of the Battery entity.
        """
        return (
            f"Battery '{self.id}' with charging power={self.quantities['max_charge_rate']} "
            ","
            f" capacity={self.quantities['capacity']}"
        )
