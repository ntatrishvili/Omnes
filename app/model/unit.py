import random
from typing import Optional
import secrets


class Unit:

    def __init__(self, id: Optional[str] = None):
        """
        Initialize the unit with an optional id.
        """
        self.id = str(id) if id is not None else secrets.token_hex(16)
        self.subunits: list[Unit] = []
        self.parent = None

    def add_unit(self, unit) -> None:
        """
        Add a subunit to the current unit.
        """
        unit.parent = self
        unit.parent_id = self.id
        self.subunits.append(unit)

    def to_pulp(self, time_set: int):
        """
        Convert the unit to a pulp representation.
        """
        res = []
        for subunit in self.subunits:
            child_objects = subunit.to_pulp(time_set)
            res.extend(child_objects)
        return res

    def __str__(self):
        """
        String representation of the unit.
        """
        subunits_str = ", ".join([str(subunit) for subunit in self.subunits])
        return f"Unit '{self.id}' containing: [{subunits_str}]"
