from typing import Optional

from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.model.device import Vector
from app.model.storage.storage import Storage


class HotWaterStorage(Storage):
    default_vector = Vector.HEAT
    default_contributes_to = "heat_balance"

    default_set_temperature: Optional[float] = None
    default_volume: Optional[float] = None

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        super().__init__(id, quantity_factory, **kwargs)
        self.create_quantity("volume", input=kwargs.pop("volume", self.default_volume))
        self.create_quantity(
            "set_temperature",
            input=kwargs.pop("set_temperature", self.default_set_temperature),
        )

    def __str__(self):
        return f"Hot water storage '{self.id}' with volume: {self.quantities['volume']} l and set temperature {self.quantities['set_temperature']} °C"
