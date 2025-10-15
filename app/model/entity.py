import secrets
from typing import Dict, Optional

from app.infra.quantity import Quantity
from app.infra.relation import Relation
from app.infra.timeseries_object_factory import (
    DefaultTimeseriesFactory,
    TimeseriesFactory,
)


class Entity:
    """
    Represents any modelled object (e.g., component, device, node) in the system.

    Entities form the core building blocks of the energy system model. Each Entity
    can have quantities (data or decision variables), hierarchical sub-entities,
    and semantic or optimization-relevant relations.

    Attributes:
        - id (str): Unique identifier.
        - quantities (Dict[str, Quantity]): Named quantities belonging to this entity.
        - sub_entities (list[Entity]): Optional nested child entities.
        - relations (list[Relation]): Constraints or rules related to this entity.
        - tags (dict): Dictionary of tags associated with this entity.
        - ts_factory (TimeseriesFactory): Used to generate time series objects in an advanced manner.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs,
    ):
        """
        Initialize the entity with an optional id.
        """
        self.id = str(id) if id is not None else secrets.token_hex(16)
        self.quantities: Dict[str, Quantity] = {}
        self.sub_entities: list[Entity] = []
        self.relations: list[Relation] = []
        self.parent = None
        self.ts_factory = ts_factory or DefaultTimeseriesFactory()
        self.relations = kwargs.pop("relations", [])
        self.tags = kwargs.pop("tags", {})
        if "input" in kwargs and "col" not in kwargs.get("input", {}):
            kwargs["input"]["col"] = self.id

    def add_sub_entity(self, entity) -> None:
        """
        Add a sub_entity to the current entity.
        """
        entity.parent = self
        entity.parent_id = self.id
        self.sub_entities.append(entity)

    def convert(self, time_set: int, new_freq: str, converter):
        """
        Delegate to a visitor for conversion.
        """
        return converter.convert_entity(self, time_set, new_freq)

    def __str__(self):
        """
        String representation of the entity.
        """
        sub_entities_str = ", ".join(
            [str(sub_entity) for sub_entity in self.sub_entities]
        )
        return f"Entity '{self.id}' containing: [{sub_entities_str}]"

    def __getattr__(self, name):
        """
        Get an attribute by name, checking parameters and quantities.
        """
        if name in self.quantities:
            return self.quantities[name]
        else:
            raise KeyError(f"'{name}' not found in parameters or quantities")

    def __dir__(self):
        """
        Extend the default dir to include parameters and quantities.
        """
        return super().__dir__() + list(self.quantities.keys())
