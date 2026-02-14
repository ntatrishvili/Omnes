from typing import Optional, Union, Dict, Any, Type, Callable

from app.infra.quantity import Quantity
from app.infra.relation import Relation
from app.infra.util import TimeSet
from app.model.entity import Entity
from app.model.model import Model


class Converter(object):
    # Default time range size when none is provided
    DEFAULT_TIME_RANGE_SIZE = 10

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

    def convert_entity(self, entity: Entity, time_set: Optional[TimeSet] = None):
        """
        Convert an entity using registered converters.
        Falls back to default entity conversion if no specialized converter exists.

        This implementation searches the entity's class MRO so that registering a
        converter for a base class (e.g., Device) will also handle subclasses
        (e.g., SpecDevice) unless a more specific converter is registered.

        Parameters
        ----------
        entity : Entity
            The entity to convert
        time_set : TimeSet, optional
            TimeSet object containing time configuration (number of steps, frequency, etc.)
        """
        # Look up specialized converter for this entity type or its base classes
        entity_type = type(entity)

        # 1) Exact match
        if entity_type in self._entity_converters:
            return self._entity_converters[entity_type](entity, time_set)

        # 2) Walk MRO (method resolution order) to find the first registered handler
        for base in entity_type.__mro__[1:]:  # skip the class itself
            if base in self._entity_converters:
                return self._entity_converters[base](entity, time_set)

        # Fall back to default entity conversion
        return self._convert_entity_default(entity, time_set)

    def _convert_entity_default(
        self, entity: Entity, time_set: Optional[TimeSet] = None, **kwargs
    ):
        """
        Default entity conversion logic.
        Subclasses must implement this method.

        Parameters
        ----------
        entity : Entity
            The entity to convert
        time_set : TimeSet, optional
            TimeSet object containing time configuration
        """
        raise NotImplementedError(
            "Subclasses must implement '_convert_entity_default'."
        )

    def convert_model(self, model: Model, time_set: Optional[TimeSet] = None, **kwargs):
        """
        Convert a model to target format.
        Subclasses must implement this method.

        Parameters
        ----------
        model : Model
            The model to convert
        time_set : TimeSet, optional
            TimeSet object containing time configuration. If None, uses model defaults.
        """
        raise NotImplementedError("Subclasses must implement 'convert_model'.")

    def convert_quantity(
        self,
        quantity: Quantity,
        name: str,
        time_set: Optional[TimeSet] = None,
    ):
        """
        Convert a quantity to target format.
        Subclasses must implement this method.

        Parameters
        ----------
        quantity : Quantity
            The quantity to convert
        name : str
            Name for the quantity
        time_set : TimeSet, optional
            TimeSet object containing time configuration
        """
        raise NotImplementedError("Subclasses must implement 'convert_quantity'.")

    def convert_relation(
        self,
        relation: Relation,
        entity_variables: Dict[str, Any],
        time_set: Optional[TimeSet] = None,
    ):
        """
        Convert a relation to target format.
        Subclasses must implement this method.

        Parameters
        ----------
        relation : Relation
            The relation to convert
        entity_variables : Dict[str, Any]
            Dictionary mapping entity IDs to their variables
        time_set : TimeSet, optional
            The TimeSet object containing full time information
        """
        raise NotImplementedError("Subclasses must implement 'convert_relation'.")

    def convert_literal(
        self,
        literal,
        value,
        t: int,
        time_set: Optional[TimeSet] = None,
    ):
        """
        Convert a literal value (returns the value as-is).

        Parameters
        ----------
        literal : Literal
            The literal object
        value : Any
            The literal value
        t : int
            Current time step
        time_set : TimeSet, optional
            TimeSet object (unused for literals)
        """
        return value

    def convert_time_condition_expression(
        self,
        time_condition_expression,
        entity: str,
        condition: str,
        between_time_index,
        t: int,
    ):
        """
        Convert a time condition expression.
        Subclasses must implement this method.

        Parameters
        ----------
        time_condition_expression : TimeConditionExpression
            The time condition expression object
        entity : str
            The entity ID
        condition : str
            The condition type ('enabled', 'disabled', etc.)
        between_time_index : array-like
            Index of time steps that fall within the time window
        t : int
            Current time step
        """
        raise NotImplementedError(
            "Subclasses must implement 'convert_time_condition_expression'."
        )
