from typing import Optional

from app.infra.parameter import Parameter
from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.infra.timeseries_object import TimeseriesObject
from app.model.grid_component.connector import Connector


class Line(Connector):
    """Transmission or distribution line between two buses.

    Attributes:
        default_line_length: Default physical length of the line.
        default_resistance: Default series resistance.
        default_reactance: Default series reactance.
        default_max_current: Default current limit.
        default_capacitance: Default shunt capacitance.
        current: Line current time series.
        line_length: Physical line length.
        resistance: Series resistance.
        reactance: Series reactance.
        max_current: Current limit.
        capacitance: Shunt capacitance.
    """

    default_line_length: Optional[float] = None
    default_resistance: Optional[float] = None
    default_reactance: Optional[float] = None
    default_max_current: Optional[float] = None
    default_capacitance: Optional[float] = None
    QUANTITY_NAMES = (
        "line_length",
        "resistance",
        "reactance",
        "max_current",
        "capacitance",
    )

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)
        self.create_quantity(
            "current", **kwargs.get("current", {}), default_type=TimeseriesObject
        )
        for quantity_name in self.QUANTITY_NAMES:
            self.create_quantity(
                quantity_name,
                input=kwargs.pop(
                    quantity_name, getattr(self, f"default_{quantity_name}")
                ),
                default_type=Parameter,
            )

    def __str__(self):
        """
        String representation of the Line entity.
        """
        return f"Line '{self.id}' {self.from_bus}--{self.to_bus} with length {self.line_length}"
