from typing import Optional
import pandas as pd
import pulp

from app.conversion.convert_optimization import create_empty_pulp_var


class Slack:
    def __init__(self, id: Optional[str] = None):
        self.id = id if id else "slack"
        self.flow_in = pd.DataFrame()
        self.flow_out = pd.DataFrame()

    @staticmethod
    def get_flow_in_pulp_empty(self, time_set: int) -> pulp.LpVariable:
        """
        Energy injected into the household by the slack
        """
        return create_empty_pulp_var("P_slack_in", time_set)

    @staticmethod
    def get_flow_out_pulp_empty(self, time_set: int) -> pulp.LpVariable:
        """
        Energy injected into the grid by the household
        #"""
        return create_empty_pulp_var("P_slack_out", time_set)

    def __str__(self):
        """
        String representation of the Slack unit.
        """
        return f"Slack '{self.id}' with flow_in_sum={self.flow_in.sum()},flow_out_sum={self.flow_out.sum()}"
