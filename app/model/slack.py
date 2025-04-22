from typing import Optional
import pandas as pd

from app.infra.util import create_empty_pulp_var
from app.model.unit import Unit
from app.model.timeseries_object import TimeseriesObject

class Slack(Unit):
    def __init__(self, id: Optional[str] = None):
        self.id = id if id else "slack"
        self.flow_in = TimeseriesObject()
        self.flow_out = TimeseriesObject()

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

    def get_flow_in(self):
        """
        Get the flow in of the slack unit.
        """
        return self.flow_in.get_data()

    def get_flow_out(self):
        """
        Get the flow in of the slack unit.
        """
        return self.flow_out.get_data()
    
    def __str__(self):
        """
        String representation of the Slack unit.
        """
        flow_in_sum = self.get_flow_in().sum() if not self.get_flow_in().empty else 0
        flow_out_sum = self.get_flow_out().sum() if not self.get_flow_out().empty else 0
        return f"Slack '{self.id}' with flow_in_sum = {flow_in_sum},flow_out_sum = {flow_out_sum}"
