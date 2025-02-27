import pandas as pd


class Battery:
    def __init__(
        self,
        injection: pd.DataFrame,
        withdrawal: pd.DataFrame,
        state_of_charge: pd.DataFrame,
    ):
        self.injection = injection
        self.withdrawal = withdrawal
        self.state_of_charge = state_of_charge
