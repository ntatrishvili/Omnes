from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import (
    DefaultTimeseriesFactory,
    TimeseriesFactory,
)
from app.model.entity import Entity


class GridComponent(Entity):
    """Represents both the physical and abstract elements of an electrical grid, like transformers, buses, nodes and
    power lines."""

    default_phase_count: Optional[int] = 3
    default_phase: Optional[str] = None

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.check_kwargs(**kwargs)
        self.phase_count = kwargs.pop("phase_count", self.default_phase_count)
        self.phase = kwargs.pop(
            "phase", self.default_phase if self.phase_count == 3 else None
        )

    @classmethod
    def check_kwargs(cls, **kwargs):
        phase_count = kwargs.get("phase_count", cls.default_phase_count)
        phase = kwargs.get("phase", cls.default_phase if phase_count == 3 else None)
        if phase_count not in (1, 3):
            raise ValueError(
                "Bus phase count set incorrectly. Phase count must be 1 or 3."
            )
        if phase is None and phase_count == 1:
            raise ValueError("Phase must be set for single-phase buses.")
