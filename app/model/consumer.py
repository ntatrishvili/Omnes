import pandas as pd
from .unit import Unit
from app.infra.util import fill_df


class Consumer(Unit):
    def __init__(self):
        self.consumption = fill_df("consumption")

    def get_consumption(self) -> pd.DataFrame:
        return self.consumption
