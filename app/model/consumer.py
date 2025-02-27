import pandas as pd

from .unit import Unit


class Consumer(Unit):
    def __init__(self, consumption: pd.DataFrame):
        self.consumption = consumption
