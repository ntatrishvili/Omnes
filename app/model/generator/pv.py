from typing import Optional

from app.infra.timeseries_object_factory import TimeseriesFactory
from app.model.entity import Entity
from app.model.generator.generator import Vector


class PV(Entity):
    default_vector = Vector.ELECTRICITY
    default_contributes_to = "electric_power_balance"

    def __init__(self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs: object):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.quantities.update({"p_pv": self.ts_factory.create("p_pv", **kwargs)})

    def __str__(self):
        """
        String representation of the PV entity.
        """
        production_sum = self["p_pv"].sum() if not self["p_pv"].empty else 0
        return f"PV '{self.id}' with production sum = {production_sum}"
