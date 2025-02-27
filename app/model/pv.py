import pandas as pd

from .unit import Unit


class PV(Unit):
    def __init__(self, production: pd.DataFrame):
        self.production = production
