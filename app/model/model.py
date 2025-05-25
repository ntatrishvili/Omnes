import pandas as pd
from typing import Optional

from app.infra.util import flatten, get_input_path
from app.model.timeseries_object import TimeseriesObject
from app.model.unit import Unit
from app.model.battery import Battery
from app.model.consumer import Consumer
from app.model.pv import PV
from app.model.slack import Slack


class Model:
    def __init__(
        self,
        identifier: Optional[str] = None,
        time_set: int = 0,
        frequency: str = "15min",
    ):
        """
        Initialize the model with an optional name
        """
        self.identifier: str = identifier if identifier else "model"
        self.time_set: int = time_set
        self.frequency: str = frequency
        self.units: list[Unit] = []

    def add_unit(self, unit: Unit):
        self.units.append(unit)

    @classmethod
    def build(cls, config: dict, time_set: int, frequency: str):
        """
        Build the model from a configuration dictionary
        """
        model = cls("model")
        model.time_set = time_set
        model.frequency = frequency
        for unit_name, content in config.items():
            unit = Unit(unit_name)
            for pv_id, info in content["pvs"].items():
                pv = PV(id=pv_id)
                pv.timeseries["p_pv"] = TimeseriesObject.read(
                    get_input_path(info["filename"]), pv_id
                ).resample_to(model.frequency)
                unit.add_unit(pv)
            for cs_id, info in content["consumers"].items():
                cs = Consumer(id=cs_id)
                cs.timeseries["p_cons"] = TimeseriesObject.read(
                    get_input_path(info["filename"]), cs_id
                ).resample_to(model.frequency)
                unit.add_unit(cs)
            for b_id, info in content["batteries"].items():
                b = Battery(b_id)
                b.max_power = info["nominal_power"]
                b.capacity = info["capacity"]
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
            pulp_vars.extend(unit.to_pulp(self.time_set, self.frequency))
        pulp_vars = flatten(pulp_vars)
        pulp_vars = {k: v for d in pulp_vars for k, v in d.items()}
        pulp_vars["time_set"] = range(self.time_set)
        return pulp_vars

    def __str__(self):
        """
        String representation of the model
        """
        return "\n".join([str(unit) for unit in self.units])
