import pandas as pd
from typing import Optional

from app.model.unit import Unit
from app.model.battery import Battery
from app.model.consumer import Consumer
from app.model.pv import PV
from app.model.slack import Slack


class Model:
    def __init__(self, id: Optional[str] = None):
        """
        Initialize the model with an optional name
        """
        self.id: str = id if id else "model"
        self.units: list[Unit] = []

    def add_unit(self, unit: Unit):
        self.units.append(unit)


    @classmethod
    def build(cls, config: dict):
        """
        Build the model from a configuration dictionary
        """
        model = cls(id="model")
        df = pd.read_csv("/home/nia/Documents/BME/Semester6/thesis/Omnes/data/input.csv")
        for unit_name, content in config.items():
            unit = Unit(unit_name)
            for generator in content['pvs']:
                pv = PV(id=generator)
                pv.production = df["production"].copy()
                pv.production.index = pd.to_datetime(df["timestamp"])
                unit.add_unit(pv)
            for load in content['consumers']:
                cs = Consumer(id=load)
                cs.consumption = df["consumption"].copy()
                cs.consumption.index = pd.to_datetime(df["timestamp"])
                unit.add_unit(cs)
            for battery in content['batteries']:
                unit.add_unit(Battery(id=battery))
            model.add_unit(unit)
        model.add_unit(Slack(id="slack"))
        return model
    
    def __str__(self):
        """
        String representation of the model
        """
        return "\n".join([unit.id for unit in self.units])