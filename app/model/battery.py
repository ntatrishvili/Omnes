from typing import Optional

from .entity import Entity
from .timeseries_object_factory import TimeseriesFactory


class Battery(Entity):
    def __init__(self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.parameters = {"max_power": kwargs.get("max_power", 0), "capacity": kwargs.get("capacity", 0), }
        self.quantities = {"p_bess_in": self.ts_factory.create("p_bess_in", **kwargs),
                           "p_bess_out": self.ts_factory.create("p_bess_out", **kwargs),
                           "e_bess_stor": self.ts_factory.create("e_bess_stor", **kwargs), }

    def __str__(self):
        """
        String representation of the Battery entity.
        """
        return (f"Battery '{self.id}' with max_power={self.parameters['max_power']} "","
                f" capacity={self.parameters['capacity']}")
