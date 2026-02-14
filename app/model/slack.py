from typing import Optional

from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.infra.timeseries_object import TimeseriesObject
from app.model.device import Device


class Slack(Device):
    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)
        self.create_quantity(
            "p_in", **kwargs.get("p_in", {}), default_type=TimeseriesObject
        )
        self.create_quantity(
            "p_out", **kwargs.get("p_out", {}), default_type=TimeseriesObject
        )

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
