from typing import Optional

from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.model.generator.generator import Generator


class Wind(Generator):
    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs: object,
    ):
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)

    def __str__(self):
        production_sum = self.p_out.sum() if not self.p_out.empty else 0
        return f"Wind turbine '{self.id}' with production sum = {production_sum}"
