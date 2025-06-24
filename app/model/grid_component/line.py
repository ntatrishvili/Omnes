from typing import Optional

from app.model.grid_component.grid_component import GridComponent
from app.model.timeseries_object_factory import TimeseriesFactory


class Line(GridComponent):
    def __init__(
        self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.quantities = {
            "current": self.ts_factory.create("current", **kwargs),
            "phase_count": kwargs.get("phase_count", 1),
            "phase": kwargs.get("phase", "A"),
            "from_bus": kwargs.get("from_bus"),
            "to_bus": kwargs.get("to_bus"),
            "line_length": kwargs.get("line_length"),
            "reactance": kwargs.get("reactance"),
            "resistance": kwargs.get("resistance"),
        }

    def __str__(self):
        """
        String representation of the Bus entity.
        """
        return f"Line '{self.id}' {self['from_bus']}--{self['to_bus']} with length {self['line_length']}"
