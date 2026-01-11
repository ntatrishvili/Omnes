from typing import Optional

from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.model.device import Device


class Storage(Device):
    default_capacity: Optional[float] = None
    default_max_charge_rate: Optional[float] = None
    default_max_discharge_rate: Optional[float] = None
    default_charge_efficiency: Optional[float] = 1.0
    default_discharge_efficiency: Optional[float] = 1.0
    default_storage_efficiency: Optional[float] = 1.0

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        super().__init__(id, quantity_factory, **kwargs)
        self.create_quantity("p_in", **kwargs.get("p_in", {}))
        self.create_quantity("p_out", **kwargs.get("p_out", {}))
        self.create_quantity("e_stor", **kwargs.get("e_stor", {}))
        for quantity_name in (
            "capacity",
            "max_charge_rate",
            "max_discharge_rate",
            "charge_efficiency",
            "discharge_efficiency",
            "storage_efficiency",
        ):
            self.create_quantity(
                quantity_name,
                input=kwargs.pop(
                    quantity_name, getattr(self, f"default_{quantity_name}")
                ),
            )
