from typing import Optional
import pandas as pd

from app.infra.util import create_empty_pulp_var
from app.model.entity import Entity
from app.model.timeseries_object import TimeseriesObject
from app.model.timeseries_object_factory import TimeseriesFactory


class Slack(Entity):
    def __init__(self, id: Optional[str] = None, ts_factory: TimeseriesFactory=None, **kwargs):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.id = id if id else "slack"
        self.quantities = {
            "p_slack_in": self.ts_factory.create("p_slack_in", **kwargs),
            "p_slack_out": self.ts_factory.create("p_slack_out", **kwargs),
        }

    def get_flow_in(self):
        """
        Get the flow in of the slack entity.
        """
        return self.quantities["p_slack_in"].to_df()

    def get_flow_out(self):
        """
        Get the flow in of the slack entity.
        """
        return self.quantities["p_slack_out"].to_df()

    def __str__(self):
        """
        String representation of the Slack entity.
        """
        flow_in_sum = self.get_flow_in().sum() if not self.get_flow_in().empty else 0
        flow_out_sum = self.get_flow_out().sum() if not self.get_flow_out().empty else 0
        return (
            f"Slack '{self.id}' with flow_in_sum = {flow_in_sum},flow_out_sum ="
            f" {flow_out_sum}"
        )
