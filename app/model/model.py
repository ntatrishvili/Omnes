import pandas as pd
from typing import Optional

from app.infra.util import flatten
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
        self.time_set: int = 0
        self.units: list[Unit] = []

    def add_unit(self, unit: Unit):
        self.units.append(unit)

    @classmethod
    def build(cls, config: dict, input_path: str):
        """
        Build the model from a configuration dictionary
        """
        model = cls(id="model")
        df = pd.read_csv(input_path)
        model.time_set = len(df)
        for unit_name, content in config.items():
            unit = Unit(unit_name)
            for pv_id in content["pvs"]:
                pv = PV(id=pv_id)
                pv.production = df[pv_id].copy()
                pv.production.index = pd.to_datetime(df["timestamp"])
                unit.add_unit(pv)
            for cs_id in content["consumers"]:
                cs = Consumer(id=cs_id)
                cs.consumption = df[cs_id].copy()
                cs.consumption.index = pd.to_datetime(df["timestamp"])
                unit.add_unit(cs)
            for battery in content["batteries"]:
                b = Battery(battery["id"])
                b.max_power = battery["nominal_power"]
                b.capacity = battery["capacity"]
                unit.add_unit(b)
            model.add_unit(unit)
        model.add_unit(Slack(id="slack"))
        return model

    def to_pulp(self):
        """
        Convert the model to pulp variables
        """
        pulp_vars = []
        for unit in self.units:
            pulp_vars.extend(unit.to_pulp(self.time_set))
        pulp_vars = flatten(pulp_vars)
        pulp_vars = {k: v for d in pulp_vars for k, v in d.items()}
        pulp_vars["time_set"] = range(self.time_set)
        return pulp_vars

    def __str__(self):
        """
        String representation of the model
        """
        return "\n".join([str(unit) for unit in self.units])
