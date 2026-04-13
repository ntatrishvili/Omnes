from abc import abstractmethod
from typing import Optional, Union, Dict, Any, Type, Callable

from app.infra.quantity import Quantity
from app.infra.relation import Relation
from app.infra.util import TimeSet
from app.infra.logging_setup import get_logger
from app.model.entity import Entity
from app.model.model import Model
from app.conversion.validation_utils import extract_effective_time_properties

log = get_logger(__name__)


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
        Convert an entity using registered converters with recursive sub-entity traversal.
        
        This is the TEMPLATE METHOD that defines the traversal algorithm:
        1. Convert current entity's quantities/structure
        2. Recursively convert all sub-entities
        3. Merge results
        
        Subclasses only implement _convert_entity_default() for leaf operations.

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
        log.debug(f"Converting entity '{entity.id}' with quantities: {list(entity.quantities.keys())}")
        # Look up specialized converter for this entity type or its base classes
        entity_type = type(entity)

        # Convert this entity's quantities/structure
        if entity_type in self._entity_converters:
            result = self._entity_converters[entity_type](entity, time_set)
        else:
            # 2) Walk MRO (method resolution order) to find the first registered handler
            handler_found = False
            for base in entity_type.__mro__[1:]:  # skip the class itself
                log.debug(f"Checking for converter for base class '{base.__name__}' of entity '{entity.id}'")
                if base in self._entity_converters:
                    result = self._entity_converters[base](entity, time_set)
                    handler_found = True
                    break
            
            if not handler_found:
                result = self._convert_entity_default(entity, time_set)
        
        # Template: Recursively traverse sub-entities (same for ALL converters)
        if hasattr(entity, 'sub_entities'):
            for _, sub_entity in entity.sub_entities.items():
                log.debug(f"Recursively converting '{entity.id}' sub-entity '{sub_entity.id}'")
                sub_result = self.convert_entity(sub_entity, time_set)
                # Merge results - subclass can override _merge_results if needed
                result = self._merge_entity_results(result, sub_result)
        
        return result

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

    def _merge_entity_results(self, parent_result: Any, child_result: Any) -> Any:
        """
        Merge results from parent entity and child sub-entity conversions.
        
        Default implementation: if both are dicts, merge them; otherwise return parent.
        Subclasses can override for custom merging logic.
        
        Parameters
        ----------
        parent_result : Any
            Result from converting the parent entity
        child_result : Any
            Result from converting a child sub-entity
            
        Returns
        -------
        Any
            Merged result
        """
        if isinstance(parent_result, dict) and isinstance(child_result, dict):
            parent_result.update(child_result)
            return parent_result
        # For non-dict results (e.g., pandapower side effects), just return parent
        return parent_result

    def convert_model(self, model: Model, time_set: Optional[TimeSet] = None, **kwargs):
        """
        TEMPLATE METHOD: Convert a model to target format.
        
        This method defines the standard conversion algorithm with customizable hooks:
        1. _prepare_conversion() - Setup state, extract time_set
        2. _convert_entities() - Convert all model entities
        3. _post_process_conversion() - Format-specific post-processing
        4. _finalize_result() - Prepare final return value
        
        Subclasses override the hook methods (_prepare_conversion, etc.) 
        rather than this method.

        Parameters
        ----------
        model : Model
            The model to convert
        time_set : TimeSet, optional
            TimeSet object containing time configuration. If None, uses model defaults.
        **kwargs
            Additional converter-specific options
            
        Returns
        -------
        Any
            Converted model in target format (dict, pandapowerNet, etc.)
        """
        # Phase 1: Prepare conversion (extract time_set, reset state, etc.)
        effective_time_set, context = self._prepare_conversion(model, time_set, **kwargs)
        
        # Phase 2: Convert all entities
        result = self._convert_entities(model, effective_time_set, context, **kwargs)
        
        # Phase 3: Post-processing (relations, constraints, metadata, etc.)
        result = self._post_process_conversion(model, result, effective_time_set, context, **kwargs)
        
        # Phase 4: Finalize and return
        return self._finalize_result(result, effective_time_set, context)
    
    def _prepare_conversion(
        self, model: Model, time_set: Optional[TimeSet], **kwargs
    ) -> tuple[TimeSet, Dict[str, Any]]:
        """
        Prepare for conversion: extract effective time_set and initialize state.
        
        Subclasses can override to add custom initialization (e.g., reset network state).
        
        Parameters
        ----------
        model : Model
            The model to convert
        time_set : TimeSet, optional
            TimeSet provided by caller (may be None)
        **kwargs
            Additional options
            
        Returns
        -------
        tuple[TimeSet, Dict[str, Any]]
            (effective_time_set, conversion_context)
            - effective_time_set: The resolved TimeSet to use
            - conversion_context: Dict for storing state during conversion
        """
        effective_time_set = extract_effective_time_properties(model, time_set)
        context = {}  # Empty context by default
        return effective_time_set, context
    
    def _convert_entities(
        self, 
        model: Model, 
        time_set: TimeSet, 
        context: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Convert all model entities.
        
        Default implementation: Loop through entities and accumulate results.
        Subclasses can override for custom entity handling.
        
        Parameters
        ----------
        model : Model
            The model to convert
        time_set : TimeSet
            Effective TimeSet to use
        context : Dict[str, Any]
            Conversion context from _prepare_conversion
        **kwargs
            Additional options (e.g., skip_entities)
            
        Returns
        -------
        Any
            Accumulated entity conversion results (format depends on subclass)
        """
        skip_entities = kwargs.get("skip_entities", set())
        result = {}  # Default: accumulate in dict
        
        for _, entity in model.entities.items():
            if type(entity) in skip_entities:
                continue
            entity_result = self.convert_entity(entity, time_set)
            
            # Merge entity results
            if isinstance(result, dict) and isinstance(entity_result, dict):
                log.debug(f"Merging results from entity '{entity.id}' into overall result.")
                result.update(entity_result)
            # For non-dict results (e.g., side effects), subclass should override
        
        return result
    
    def _post_process_conversion(
        self,
        model: Model,
        result: Any,
        time_set: TimeSet,
        context: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Post-processing after entity conversion.
        
        Subclasses override to add:
        - Relation/constraint conversion
        - Metadata attachment
        - Format-specific finalization
        
        Parameters
        ----------
        model : Model
            The model being converted
        result : Any
            Results from _convert_entities
        time_set : TimeSet
            Effective TimeSet
        context : Dict[str, Any]
            Conversion context
        **kwargs
            Additional options
            
        Returns
        -------
        Any
            Post-processed result
        """
        # Default: no post-processing
        return result
    
    def _finalize_result(
        self, 
        result: Any, 
        time_set: TimeSet, 
        context: Dict[str, Any]
    ) -> Any:
        """
        Finalize the conversion result before returning.
        
        Subclasses override to:
        - Add metadata (e.g., time_set info)
        - Transform result format
        - Clean up state
        
        Parameters
        ----------
        result : Any
            Post-processed result
        time_set : TimeSet
            Effective TimeSet
        context : Dict[str, Any]
            Conversion context
            
        Returns
        -------
        Any
            Final result to return from convert_model()
        """
        # Default: add time_set to dict results
        if isinstance(result, dict):
            result["time_set"] = time_set
        return result

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

    @abstractmethod
    def convert_back(self, result_dict: Dict[str, Any], model: Model, **kwargs) -> None:
        """
        Generic reverse conversion from results to model.
        Subclasses can override for specialized behavior.
        """
        ...
