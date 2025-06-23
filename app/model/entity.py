import secrets
from typing import Optional, Dict

from app.conversion.converter import Converter
from app.conversion.pulp_converter import PulpConverter
from app.model.quantity import Quantity
from app.model.relation import Relation
from app.model.timeseries_object_factory import (
    TimeseriesFactory,
    DefaultTimeseriesFactory,
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
        - ts_factory (TimeseriesFactory): Used to generate time series objects in an advanced manner.
    """

    def __init__(
        self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
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

    def add_sub_entity(self, entity) -> None:
        """
        Add a sub_entity to the current entity.
        """
        entity.parent = self
        entity.parent_id = self.id
        self.sub_entities.append(entity)

    def to_pulp(
        self, time_set: int, new_freq: str, converter: Optional[Converter] = None
    ):
        """
        Delegate to a visitor for pulp conversion.
        """
        converter = converter or PulpConverter()
        return converter.convert(self, time_set, new_freq)

    def __str__(self):
        """
        String representation of the entity.
        """
        sub_entities_str = ", ".join(
            [str(sub_entity) for sub_entity in self.sub_entities]
        )
        return f"Unit '{self.id}' containing: [{sub_entities_str}]"

    def __getitem__(self, item):
        """
        Allows direct access to quantities via `entity["name"]`
        """
        if item in self.quantities:
            return self.quantities[item]
        else:
            raise KeyError(f"'{item}' not found in parameters or quantities")
