from typing import Optional

from app.infra.timeseries_object_factory import TimeseriesFactory
from app.model.entity import Entity


class Slack(Entity):
    def __init__(
        self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.id = id if id else "slack"
        self.quantities.update({
            "p_slack_in": self.ts_factory.create("p_slack_in", **kwargs),
            "p_slack_out": self.ts_factory.create("p_slack_out", **kwargs),
        })

    def __str__(self):
        """
        String representation of the Slack entity.
        """
        flow_in_sum = self["p_slack_in"].sum() if not self["p_slack_in"].empty else 0
        flow_out_sum = self["p_slack_out"].sum() if not self["p_slack_out"].empty else 0
        return (
            f"Slack '{self.id}' with flow_in_sum = {flow_in_sum},flow_out_sum ="
            f" {flow_out_sum}"
        )
