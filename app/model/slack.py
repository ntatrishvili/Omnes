from typing import Optional
import pandas as pd

from app.infra.util import create_empty_pulp_var
from app.model.unit import Unit
from app.model.timeseries_object import TimeseriesObject


class Slack(Unit):
    def __init__(self, id: Optional[str] = None):
        self.id = id if id else "slack"
        self.timeseries = {
            "p_slack_in": TimeseriesObject(),
            "p_slack_out": TimeseriesObject(),
        }

    def get_flow_in(self):
        """
        Get the flow in of the slack unit.
        """
        return self.timeseries["p_slack_in"].to_df()

    def get_flow_out(self):
        """
        Get the flow in of the slack unit.
        """
        return self.timeseries["p_slack_out"].to_df()

    def __str__(self):
        """
        String representation of the Slack unit.
        """
        flow_in_sum = self.get_flow_in().sum() if not self.get_flow_in().empty else 0
        flow_out_sum = self.get_flow_out().sum() if not self.get_flow_out().empty else 0
        return (
            f"Slack '{self.id}' with flow_in_sum = {flow_in_sum},flow_out_sum ="
            f" {flow_out_sum}"
        )
