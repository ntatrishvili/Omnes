from typing import Optional

from app.infra.quantity import Constant
from app.infra.timeseries_object_factory import TimeseriesFactory
from app.model.entity import Entity


class GridComponent(Entity):
    """Represents both the physical and abstract elements of an electrical grid, like transformers, buses, nodes and
    power lines."""

    default_phase_count: Optional[int] = 3
    default_phase: Optional[str] = "A"

    def __init__(
        self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.check_kwargs(**kwargs)
        self.quantities.update(
            {
                "phase_count": Constant(
                    value=kwargs.get("phase_count", self.default_phase_count)
                ),
                "phase": Constant(value=kwargs.get("phase", self.default_phase)),
            }
        )

    @staticmethod
    def check_kwargs(**kwargs):
        phase_count = kwargs.get("phase_count", 1)
        phase = kwargs.get("phase", None)
        if phase_count not in (1, 3):
            raise ValueError(
                "Bus phase count set incorrectly. Phase count must be 1 or 3."
            )
        if phase is None and phase_count == 1:
            raise ValueError("Phase must be set for single-phase buses.")
