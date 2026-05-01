"""Base device types shared by model entities."""

from enum import Enum
from typing import Optional

from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.model.entity import Entity


class Vector(Enum):
    ELECTRICITY = "electricity"
    HEAT = "heat"
    MATERIAL = "material"
    INVALID = "invalid"


class Device(Entity):
    """Base class for model devices attached to a bus.

    Attributes:
        default_vector: Default energy vector used for the device.
        default_contributes_to: Default balance key the device contributes to.
        bus: Bus identifier or bus entity assigned to the device.
        tags: Free-form metadata tags, including vector information.
    """

    _quantity_excludes = ["default_vector", "default_contributes_to"]

    default_vector: Optional[Vector] = Vector.INVALID
    default_contributes_to: Optional[str] = None

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        super().__init__(id, quantity_factory, **kwargs)
        self.bus = kwargs.pop("bus", "bus")
        if self.bus is None:
            raise ValueError(f"No bus specified for device '{self.id}'")
        tags = kwargs.get("tags", {})
        self.tags = {
            "vector": tags.pop("vector", self.default_vector),
            "contributes_to": tags.pop("contributes_to", self.default_contributes_to),
        }
        self.tags.update(tags)

    def update_tags(self, **tags):
        self.tags.update(**tags)
