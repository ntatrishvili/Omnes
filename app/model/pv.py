from typing import Optional

import pandas as pd

from .entity import Entity
from .timeseries_object_factory import TimeseriesFactory


class PV(Entity):
    def __init__(self, id: Optional[str] = None, ts_factory: TimeseriesFactory=None, **kwargs):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.quantities = {
            "p_pv": self.ts_factory.create("p_pv", **kwargs)
        }

    def get_production(self) -> pd.DataFrame:
        return self.quantities["p_pv"].to_df()

    def __str__(self):
        """
        String representation of the PV entity.
        """
        production_sum = (self.get_production().sum() if not self.get_production().empty else 0)
        return f"PV '{self.id}' with production sum = {production_sum}"
