from typing import Optional

from app.model.entity import Entity
from app.model.timeseries_object_factory import TimeseriesFactory


class GridComponent(Entity):
    """Represents the physical and abstract elements of the electrical grid."""

    def __init__(
        self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.check_kwargs(**kwargs)

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
