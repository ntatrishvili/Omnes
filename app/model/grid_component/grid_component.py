from typing import Optional

from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.model.entity import Entity


class GridComponent(Entity):
    """Represents both the physical and abstract elements of an electrical grid, like transformers, buses, nodes and
    power lines."""

    _quantity_excludes = ["default_phase"]

    default_phase: Optional[str | int] = 3

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs
    ):
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)
        self.check_kwargs(**kwargs)
        self.phase = kwargs.pop("phase", self.default_phase)
        # Only convert phase to int if it's "3", otherwise keep as str
        if self.phase == "3":
            self.phase = int(self.phase)
        self.coordinates = kwargs.pop("coordinates", [])

    @classmethod
    def check_kwargs(cls, **kwargs):
        phase = kwargs.get("phase", cls.default_phase)
        if phase not in ("A", "B", "C", 3, "3"):
            raise ValueError(f"Phase must be 'A', 'B', 'C', or 3, but got {phase!r}")
