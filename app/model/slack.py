from typing import Optional
import pandas as pd

from app.infra.util import create_empty_pulp_var
from app.model.unit import Unit


class Slack(Unit):
    def __init__(self, id: Optional[str] = None):
        self.id = id if id else "slack"
        self.flow_in = pd.DataFrame()
        self.flow_out = pd.DataFrame()

    @staticmethod
    def get_flow_in_pulp_empty(time_set: int):
        """
        Energy injected into the household by the slack
        """
        return {"p_slack_in": create_empty_pulp_var("p_slack_in", time_set)}

    @staticmethod
    def get_flow_out_pulp_empty(time_set: int):
        """
        Energy injected into the grid by the household
        #"""
        return {"p_slack_out": create_empty_pulp_var("p_slack_out", time_set)}

    def to_pulp(self, time_set: int):
        """
        Convert the Slack unit to pulp variables.
        """
        flow_in = self.get_flow_in_pulp_empty(time_set)
        flow_out = self.get_flow_out_pulp_empty(time_set)
        return [flow_in, flow_out]

    def __str__(self):
        """
        String representation of the Slack unit.
        """
        flow_in_sum = self.flow_in.sum() if not self.flow_in.empty else 0
        flow_out_sum = self.flow_out.sum() if not self.flow_out.empty else 0
        return f"Slack '{self.id}' with flow_in_sum = {flow_in_sum},flow_out_sum = {flow_out_sum}"
