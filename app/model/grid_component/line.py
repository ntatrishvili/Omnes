from typing import Optional

from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.model.grid_component.connector import Connector


class Line(Connector):
    default_line_length: Optional[float] = None
    default_resistance: Optional[float] = None
    default_reactance: Optional[float] = None
    default_max_current: Optional[float] = None
    default_capacitance: Optional[float] = None

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)
        self.create_quantity("current", **kwargs.get("current", {}))
        for quantity_name in (
            "line_length",
            "resistance",
            "reactance",
            "max_current",
            "capacitance",
        ):
            self.create_quantity(
                quantity_name,
                input=kwargs.pop(
                    quantity_name, getattr(self, f"default_{quantity_name}")
                ),
            )

    def __str__(self):
        """
        String representation of the Line entity.
        """
        return f"Line '{self.id}' {self.from_bus}--{self.to_bus} with length {self.line_length}"
