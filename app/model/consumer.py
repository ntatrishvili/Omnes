from typing import Optional
import pandas as pd
from .unit import Unit
from app.infra.util import fill_df


class Consumer(Unit):
    def __init__(self, id: Optional[str] = None):
        super().__init__(id)
        self.consumption = None

    def get_consumption(self) -> pd.DataFrame:
        return self.consumption

    def __str__(self):
        """
        String representation of the Consumer unit.
        """
        return f"Consumer {self.id} with consumption_sum={self.consumption.sum()}"
