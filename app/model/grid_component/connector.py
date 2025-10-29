from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import (
    DefaultTimeseriesFactory,
    TimeseriesFactory,
)
from app.model.grid_component.grid_component import GridComponent


class Connector(GridComponent):
    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs,
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.from_bus = kwargs.pop("from_bus")
        self.to_bus = kwargs.pop("to_bus")

    def __str__(self):
        """
        String representation of the Connector entity.
        """
        return f"Connector '{self.id}' {self.from_bus}--{self.to_bus}"
