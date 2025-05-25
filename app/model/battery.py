from typing import Optional, override
import pandas as pd

from .unit import Unit
from app.infra.util import create_empty_pulp_var
from app.model.timeseries_object import TimeseriesObject


class Battery(Unit):
    def __init__(self, id: Optional[str] = None):
        super().__init__(id)
        self.max_power = 0
        self.capacity = 0
        self.timeseries = {
            "p_bess_in": TimeseriesObject(),
            "p_bess_out": TimeseriesObject(),
            "e_bess_stor": TimeseriesObject(),
        }

    @staticmethod
    def get_injection_pulp_empty(time_set: int):
        """
        Input electric power of the battery
        """
        return {"p_bess_in": create_empty_pulp_var("bess_in", time_set)}

    @staticmethod
    def get_withdrawal_pulp_empty(time_set: int):
        """
        Output electric power of the battery
        """
        return {"p_bess_out": create_empty_pulp_var("bess_out", time_set)}

    @staticmethod
    def get_soc_pulp_empty(time_set: int):
        """
        Stored electric energy
        """
        return {"e_bess_stor": create_empty_pulp_var("bess_soc", time_set)}

    def get_max_power(self):
        """
        Get the maximum power of the battery.
        """
        return {"max_power_bess": self.max_power}

    def get_capacity(self):
        """
        Get the capacity of the battery.
        """
        return {"max_stored_energy_bess": self.capacity}

    def __str__(self):
        """
        String representation of the Battery unit.
        """
        return (
            f"Battery '{self.id}' with max_power={self.max_power},"
            f" capacity={self.capacity}"
        )

    @override
    def to_pulp(self, time_set: int, frequency: str):
        """
        Convert the Battery unit to a pulp variable.
        """
        pulp_vars = [
            self.get_withdrawal_pulp_empty(time_set),
            self.get_injection_pulp_empty(time_set),
            self.get_soc_pulp_empty(time_set),
            self.get_max_power(),
            self.get_capacity(),
        ]
        return pulp_vars
