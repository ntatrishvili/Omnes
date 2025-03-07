import pandas as pd

from .unit import Unit
from app.infra.util import fill_df


class PV(Unit):
    def __init__(self):
        self.production = fill_df("production")

    def get_production(self) -> pd.DataFrame:
        return self.production
