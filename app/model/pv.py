from typing import Optional
import pandas as pd

from .unit import Unit
from app.infra.util import create_empty_pulp_var


class PV(Unit):
    def __init__(self, id: Optional[str] = None):
        super().__init__(id)
        self.production = pd.DataFrame()

    def get_production(self) -> pd.DataFrame:
        return self.production

    def to_pulp(self, time_set: int):
        """
        Convert the PV unit to a pulp variable.
        """
        if self.production.empty:
            return [{"p_pv": create_empty_pulp_var("p_pv", time_set)},]
        return [{"p_pv": self.get_production()},]

    def __str__(self):
        """
        String representation of the PV unit.
        """
        production_sum = self.production.sum() if not self.production.empty else 0
        return f"PV '{self.id}' with production_sum = {production_sum}"
