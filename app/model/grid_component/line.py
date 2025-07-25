from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import (
    TimeseriesFactory,
    DefaultTimeseriesFactory,
)
from app.model.grid_component.grid_component import GridComponent


class Line(GridComponent):
    default_line_length: Optional[float] = None
    default_resistance: Optional[float] = None
    default_reactance: Optional[float] = None
    default_max_current: Optional[float] = None

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs,
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.from_bus = kwargs.pop("from_bus")
        self.to_bus = kwargs.pop("to_bus")
        self.quantities.update(
            {
                "current": self.ts_factory.create("current", **kwargs),
                "max_current": Parameter(
                    value=kwargs.pop("max_current", self.default_max_current)
                ),
                "line_length": Parameter(
                    value=kwargs.pop("line_length", self.default_line_length)
                ),
                "reactance": Parameter(
                    value=kwargs.pop("reactance", self.default_reactance)
                ),
                "resistance": Parameter(
                    value=kwargs.pop("resistance", self.default_resistance)
                ),
            }
        )

    def __str__(self):
        """
        String representation of the Bus entity.
        """
        return f"Line '{self.id}' {self.from_bus}--{self.to_bus} with length {self.line_length}"
