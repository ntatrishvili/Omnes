from typing import Optional

from app.infra.timeseries_object_factory import (
    DefaultTimeseriesFactory,
    TimeseriesFactory,
)
from app.model.device import Device
from app.model.entity import Entity


class Slack(Device):
    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs,
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.create_quantity("p_in", **kwargs)
        self.create_quantity("p_out", **kwargs)

    def __str__(self):
        """
        String representation of the Slack entity.
        """
        flow_in_sum = self.p_in.sum() if not self.p_in.empty else 0
        flow_out_sum = self.p_out.sum() if not self.p_out.empty else 0
        return (
            f"Slack '{self.id}' with flow_in_sum = {flow_in_sum},flow_out_sum ="
            f" {flow_out_sum}"
        )
