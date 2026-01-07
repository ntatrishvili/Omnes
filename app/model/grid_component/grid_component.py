from typing import Optional

from app.infra.timeseries_object_factory import (DefaultTimeseriesFactory, TimeseriesFactory, )
from app.model.entity import Entity
from app.model.util import InitOnSet


def _cast_phase(phase: Optional[str | int], default=3) -> str | int:
    if phase is None:
        return default
    try:
        return int(phase)
    except ValueError:
        return str(phase)


class GridComponent(Entity):
    """Represents both the physical and abstract elements of an electrical grid, like transformers, buses, nodes and
    power lines."""

    default_phase: Optional[str | int] = InitOnSet(_cast_phase, default=3)

    def __init__(self, id: Optional[str] = None, ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(), **kwargs):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.check_kwargs(**kwargs)
        self.phase = kwargs.pop("phase", self.default_phase)
        try:
            self.phase = int(self.phase)
        except ValueError:
            pass
        self.coordinates = kwargs.pop("coordinates", [])

    @classmethod
    def check_kwargs(cls, **kwargs):
        phase = kwargs.get("phase", cls.default_phase)
        if phase not in ("A", "B", "C", 3):
            raise ValueError("Phase must be 'A', 'B', 'C', 3")
