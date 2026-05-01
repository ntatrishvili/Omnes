from typing import Optional

from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.model.grid_component.grid_component import GridComponent


class Connector(GridComponent):
    """Grid component that connects two buses.

    Attributes:
        from_bus: Source bus identifier or bus entity.
        to_bus: Target bus identifier or bus entity.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)
        self.from_bus = kwargs.pop("from_bus")
        self.to_bus = kwargs.pop("to_bus")

    def __str__(self):
        """
        String representation of the Connector entity.
        """
        return f"Connector '{self.id}' {self.from_bus}--{self.to_bus}"
