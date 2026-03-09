from typing import Optional

from ...infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from ..device import Vector
from .storage import Storage


class Battery(Storage):
    default_vector = Vector.ELECTRICITY
    default_contributes_to = "electric_power_balance"

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)

    def __str__(self):
        """
        String representation of the Battery entity.
        """
        return (
            f"Battery '{self.id}' with charging power={self.quantities['max_charge_rate']} "
            ","
            f" capacity={self.quantities['capacity']}"
        )
