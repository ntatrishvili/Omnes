from typing import Optional

from app.infra.parameter import Parameter
from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.infra.timeseries_object import TimeseriesObject
from app.model.device import Device, Vector


class Load(Device):
    default_vector = Vector.ELECTRICITY
    default_contributes_to = "electric_power_balance"

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)
        self.create_quantity(
            "p_cons", **kwargs.get("p_cons", {}), default_type=TimeseriesObject
        )
        self.create_quantity(
            "q_cons", **kwargs.get("q_cons", {}), default_type=TimeseriesObject
        )
        self.create_quantity(
            "nominal_power",
            input=kwargs.pop("nominal_power", None),
            default_type=Parameter,
        )

    def __str__(self):
        """
        String representation of the Load entity.
        """
        consumption_sum = self.p_cons.sum() if not self.p_cons.empty else 0
        return f"Load '{self.id}' with consumption_sum={consumption_sum}"
