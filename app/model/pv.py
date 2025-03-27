from typing import Optional
import pandas as pd

from .unit import Unit
from app.infra.util import fill_df


class PV(Unit):
    def __init__(self, id: Optional[str] = None):
        super().__init__(id)
        self.production = None

    def get_production(self) -> pd.DataFrame:
        return self.production
