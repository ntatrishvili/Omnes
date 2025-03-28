from typing import Optional
import pandas as pd

from .unit import Unit
from app.infra.util import create_empty_pulp_var


class Battery(Unit):
    def __init__(self, id: Optional[str] = None):
        super().__init__(id)
        self.injection = pd.DataFrame()
        self.withdrawal = pd.DataFrame()
        self.state_of_charge = pd.DataFrame()

    @staticmethod
    def get_injection_pulp_empty(self, time_set: int):
        """
        Input electric power of the battery
        """
        return {"bess_in": create_empty_pulp_var("bess_in", time_set)}

    @staticmethod
    def get_withdrawal_pulp_empty(self, time_set: int):
        """
        Output electric power of the battery
        """
        return {"bess_out": create_empty_pulp_var("bess_out", time_set)}

    @staticmethod
    def get_soc_pulp_empty(self, time_set: int):
        """
        Stored electric energy
        """
        return {"bess_soc": create_empty_pulp_var("bess_soc", time_set)}

    def __str__(self):
        """
        String representation of the Battery unit.
        """
        injection_sum = self.injection.sum() if not self.injection.empty else 0
        withdrawal_sum = self.withdrawal.sum() if not self.withdrawal.empty else 0
        return f"Battery '{self.id}' with injection_sum={injection_sum}, withdrawal_sum={withdrawal_sum}"

    def to_pulp(self, time_set: int):
        """
        Convert the Battery unit to a pulp variable.
        """
        pulp_vars = [
            self.get_withdrawal_pulp_empty(time_set),
            self.get_injection_pulp_empty(time_set),
            self.get_soc_pulp_empty(time_set),
        ]
        return pulp_vars
