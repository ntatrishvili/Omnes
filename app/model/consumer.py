from typing import Optional

from .entity import Entity
from .timeseries_object_factory import TimeseriesFactory


class Consumer(Entity):
    def __init__(
        self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.quantities = {"p_cons": self.ts_factory.create("p_cons", **kwargs)}

    def __str__(self):
        """
        String representation of the Consumer entity.
        """
        consumption_sum = self.p_cons.sum() if not self.p_cons.empty else 0
        return f"Consumer '{self.id}' with consumption_sum={consumption_sum}"
