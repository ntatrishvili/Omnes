from typing import Optional
import pandas as pd
import pulp

from .unit import Unit
from ..conversion.convert_optimization import create_empty_pulp_var


class Battery(Unit):
    def __init__(self, id: Optional[str] = None):
        super().__init__(id)
        self.injection = pd.DataFrame()
        self.withdrawal = pd.DataFrame()
        self.state_of_charge = pd.DataFrame()

    @staticmethod
    def get_injection_pulp_empty(self, time_set: int) -> pulp.LpVariable:
        """
        Input electric power of the battery
        """
        return create_empty_pulp_var("bess_in", time_set)

    @staticmethod
    def get_withdrawal_pulp_empty(self, time_set: int) -> pulp.LpVariable:
        """
        Output electric power of the battery
        """
        return create_empty_pulp_var("bess_out", time_set)

    @staticmethod
    def get_soc_pulp_empty(self, time_set: int) -> pulp.LpVariable:
        """
        Stored electric energy
        """
        return create_empty_pulp_var("bess_soc", time_set)
