from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import (
    TimeseriesFactory,
    DefaultTimeseriesFactory,
)
from app.model.entity import Entity


class GridComponent(Entity):
    """Represents both the physical and abstract elements of an electrical grid, like transformers, buses, nodes and
    power lines."""

    default_phase_count: Optional[int] = 3
    default_phase: Optional[str] = "A"

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.check_kwargs(**kwargs)
        self.phase = kwargs.pop("phase_count", self.default_phase)
        self.phase_count = kwargs.pop("phase_count", self.default_phase_count)

    @classmethod
    def check_kwargs(cls, **kwargs):
        phase_count = kwargs.pop("phase_count", cls.default_phase_count)
        phase = kwargs.pop("phase", cls.default_phase)
        if phase_count not in (1, 3):
            raise ValueError(
                "Bus phase count set incorrectly. Phase count must be 1 or 3."
            )
        if phase is None and phase_count == 1:
            raise ValueError("Phase must be set for single-phase buses.")
