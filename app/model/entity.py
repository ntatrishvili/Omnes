import secrets
from typing import Optional

from app.infra.quantity import Quantity
from app.infra.quantity_factory import QuantityFactory, DefaultQuantityFactory
from app.infra.relation import Relation
from app.model.util import InitializingMeta


class Entity(metaclass=InitializingMeta):
    """
    Represents any modelled object (e.g., component, device, node) in the system.

    Entities form the core building blocks of the energy system model. Each Entity
    can have quantities (data or decision variables), hierarchical sub-entities,
    and semantic or optimization-relevant relations.

    Attributes:
        - id (str): Unique identifier.
        - quantities (Dict[str, Quantity]): Named quantities belonging to this entity.
        - sub_entities (dict[str, Entity]): Optional nested child entities.
        - relations (list[Relation]): Constraints or rules related to this entity.
        - tags (dict): Dictionary of tags associated with this entity.
        - quantity_factory (QuantityFactory): Used to generate time series objects in an advanced manner.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        """
        Initialize the entity with an optional id.
        """
        self.id = str(id) if id is not None else secrets.token_hex(16)
        self.quantities: dict[str, Quantity] = {}
        self.sub_entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.parent = None
        self.quantity_factory = quantity_factory or DefaultQuantityFactory()
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
        self.sub_entities[entity.id] = entity

    def get_sub_entity(self, id):
        return self.sub_entities[id]

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
            [str(sub_entity) for sub_entity in self.sub_entities.items()]
        )
        return f"Entity '{self.id}' containing: [{sub_entities_str}]"

    def __contains__(self, item):
        return item in self.sub_entities or item in self.quantities

    def __getattr__(self, name):
        """
        Get an attribute by name, checking parameters and quantities.
        """
        try:
            quantities = object.__getattribute__(self, "quantities")
        except AttributeError:
            # If `quantities` itself doesn't exist, behave like normal attribute lookup
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        if name in quantities:
            return quantities[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __dir__(self):
        """
        Extend the default dir to include parameters and quantities.
        """
        base = super().__dir__() or []
        return base + list(self.quantities.keys())

    def create_quantity(self, name: str, **kwargs):
        self.quantities.update(
            {name: self.quantity_factory.create(name, **kwargs, entity_id=self.id)}
        )
