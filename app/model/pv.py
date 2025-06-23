from typing import Optional

from .entity import Entity
from .timeseries_object_factory import TimeseriesFactory


class PV(Entity):
    def __init__(
        self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.quantities = {"p_pv": self.ts_factory.create("p_pv", **kwargs)}

    def __str__(self):
        """
        String representation of the PV entity.
        """
        production_sum = self["p_pv"].sum() if not self["p_pv"].empty else 0
        return f"PV '{self.id}' with production sum = {production_sum}"
