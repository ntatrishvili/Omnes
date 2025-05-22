from typing import Optional
import pandas as pd

from .unit import Unit
from app.infra.util import create_empty_pulp_var
from app.model.timeseries_object import TimeseriesObject


class PV(Unit):
    def __init__(self, id: Optional[str] = None):
        super().__init__(id)
        self.timeseries = {"p_pv": TimeseriesObject()}

    def get_production(self) -> pd.DataFrame:
        return self.timeseries["p_pv"].to_df()

    def __str__(self):
        """
        String representation of the PV unit.
        """
        production_sum = (
            self.get_production().sum() if not self.get_production().empty else 0
        )
        return f"PV '{self.id}' with production_sum = {production_sum}"
