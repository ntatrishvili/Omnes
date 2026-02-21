from typing import Optional

from app.infra.parameter import Parameter
from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.infra.timeseries_object import TimeseriesObject
from app.model.device import Device, Vector


class Generator(Device):
    default_vector = Vector.ELECTRICITY
    default_contributes_to = "electric_power_balance"
    default_peak_power = 0
    default_efficiency = 0

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs
    ):
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)
        self.create_quantity(
            "p_out", **kwargs.get("p_out", {}), default_type=TimeseriesObject
        )
        self.create_quantity(
            "q_out", **kwargs.get("q_out", {}), default_type=TimeseriesObject
        )
        self.create_quantity(
            "peak_power",
            input=kwargs.pop("peak_power", self.default_peak_power),
            default_type=Parameter,
        )
        self.create_quantity(
            "efficiency",
            input=kwargs.pop("efficiency", self.default_efficiency),
            default_type=Parameter,
        )
