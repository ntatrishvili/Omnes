from typing import Optional
import pandas as pd

from .unit import Unit
from app.infra.util import create_empty_pulp_var
from app.model.timeseries_object import TimeseriesObject

class Consumer(Unit):
    def __init__(self, id: Optional[str] = None):
        super().__init__(id)
        self.consumption = TimeseriesObject()

    def get_consumption(self) -> pd.DataFrame:
        return self.consumption.get_data()

    def __str__(self):
        """
        String representation of the Consumer unit.
        """
        consumption_sum = self.get_consumption().sum() if not self.get_consumption().empty else 0
        return f"Consumer '{self.id}' with consumption_sum={consumption_sum}"

    def to_pulp(self, time_set: int):
        """
        Convert the Consumer unit to a pulp variable.
        """
        if self.consumption.get_data().empty:
            return {"p_cons": create_empty_pulp_var("p_cons", time_set)}
        return [
            {"p_cons": self.get_consumption()},
        ]
