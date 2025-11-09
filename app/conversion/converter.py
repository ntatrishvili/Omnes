from typing import Optional, Union, Dict, Any, Type, Callable

from app.infra.quantity import Quantity
from app.infra.relation import Relation
from app.model.entity import Entity
from app.model.model import Model


class Converter(object):
    # Default time set size when none is provided
    DEFAULT_TIME_SET_SIZE = 10

    def __init__(self):
        # Registry mapping entity types to their converter methods
        self._entity_converters: Dict[Type[Entity], Callable] = {}
        self._register_converters()

    def _register_converters(self):
        """
        Override this method in subclasses to register specialized converters.
        Example:
            self._entity_converters[Bus] = self._convert_bus
            self._entity_converters[Line] = self._convert_line
        """
        pass

    def convert_entity(
        self, entity: Entity, time_set: int = None, new_freq: str = None
    ):
        """
        Convert an entity using registered converters.
        Falls back to default entity conversion if no specialized converter exists.

        This implementation searches the entity's class MRO so that registering a
        converter for a base class (e.g., Device) will also handle subclasses
        (e.g., SpecDevice) unless a more specific converter is registered.
        """
        # Look up specialized converter for this entity type or its base classes
        entity_type = type(entity)

        # 1) Exact match
        if entity_type in self._entity_converters:
            return self._entity_converters[entity_type](entity, time_set, new_freq)

        # 2) Walk MRO (method resolution order) to find the first registered handler
        for base in entity_type.__mro__[1:]:  # skip the class itself
            if base in self._entity_converters:
                return self._entity_converters[base](entity, time_set, new_freq)

        # Fall back to default entity conversion
        return self._convert_entity_default(entity, time_set, new_freq)

    def _convert_entity_default(
        self, entity: Entity, time_set: int = None, new_freq: str = None, **kwargs
    ):
        """
        Default entity conversion logic.
        Subclasses must implement this method.
        """
        raise NotImplementedError(
            "Subclasses must implement '_convert_entity_default'."
        )

    def convert_model(
        self, model: Model, time_set: int = None, new_freq: str = None, **kwargs
    ):
        raise NotImplementedError("Subclasses must implement 'convert_model'.")

    def convert_quantity(
        self,
        quantity: Quantity,
        name: str,
        time_set: Optional[Union[int, range]] = None,
        freq: Optional[str] = None,
    ):
        raise NotImplementedError("Subclasses must implement 'convert_quantity'.")

    def convert_relation(
        self,
        relation: Relation,
        entity_variables: Dict[str, Any],
        time_set: Optional[Union[int, range]] = None,
        new_freq: Optional[str] = None,
    ):
        raise NotImplementedError("Subclasses must implement 'convert_relation'.")

    def convert_literal(
        self,
        literal,
        value,
        t: int,
        time_set: Optional[Union[int, range]] = None,
        new_freq: Optional[str] = None,
    ):
        """
        Convert a literal value (returns the value as-is).
        """
        return value
