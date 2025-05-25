from collections import defaultdict
from typing import Optional
import secrets

from app.model.timeseries_object import TimeseriesObject


class Unit:

    def __init__(self, id: Optional[str] = None, **kwargs):
        """
        Initialize the unit with an optional id.
        """
        self.id = str(id) if id is not None else secrets.token_hex(16)
        self.timeseries = defaultdict(TimeseriesObject)
        self.parameters = defaultdict(float)
        self.subunits: list[Unit] = []
        self.parent = None

    def add_unit(self, unit) -> None:
        """
        Add a subunit to the current unit.
        """
        unit.parent = self
        unit.parent_id = self.id
        self.subunits.append(unit)

    def to_pulp(self, time_set: int, new_freq: str):
        """
        Convert the unit to a pulp representation.
        """
        res = []
        if not hasattr(self, "subunits") or not self.subunits:
            return [
                {
                    key: ts.resample_to(new_freq).to_pulp(
                        name=key, freq=new_freq, time_set=time_set
                    )
                }
                for key, ts in self.timeseries.items()
            ]
        for subunit in self.subunits:
            child_objects = subunit.to_pulp(time_set, new_freq)
            res.extend(child_objects)
        return res

    def __str__(self):
        """
        String representation of the unit.
        """
        subunits_str = ", ".join([str(subunit) for subunit in self.subunits])
        return f"Unit '{self.id}' containing: [{subunits_str}]"
