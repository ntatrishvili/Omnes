import random
from typing import Optional


class Unit:

    def __init__(self, id: Optional[str] = None):
        """
        Initialize the unit with an optional id.
        """
        self.id = id if id is not None else str(random.randint(1, 1000000))
        self.subunits: list[Unit] = []

    def add_unit(self, unit) -> None:
        """
        Add a subunit to the current unit.
        """
        unit.parent = self
        unit.parent_id = self.id
        self.subunits.append(unit)

    def fill(self):
        """
        Fill the unit with input data.
        """
        pass  # TODO

    def __str__(self):
        """
        String representation of the unit.
        """
        subunits_str = ", ".join([str(subunit) for subunit in self.subunits])
        return f"Unit '{self.id}' containing: [{subunits_str}])"
