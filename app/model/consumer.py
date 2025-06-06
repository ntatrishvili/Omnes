from typing import Optional

import pandas as pd

from .entity import Entity
from .timeseries_object_factory import TimeseriesFactory


class Consumer(Entity):
    def __init__(self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.quantities = {"p_cons": self.ts_factory.create("p_cons", **kwargs)}

    def get_consumption(self) -> pd.DataFrame:
        return self.quantities["p_cons"].to_df()

    def __str__(self):
        """
        String representation of the Consumer entity.
        """
        consumption_sum = (self.get_consumption().sum() if not self.get_consumption().empty else 0)
        return f"Consumer '{self.id}' with consumption_sum={consumption_sum}"
