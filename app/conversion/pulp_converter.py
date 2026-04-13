from typing import Any, Dict, List, Optional, Union

import pulp

from app.conversion.conversion_utils import (
    create_empty_pulp_var,
    handle_arithmetic_operator,
    handle_comparison_operator,
    is_arithmetic_operator,
    is_comparison_operator,
)
from app.conversion.converter import Converter
from app.conversion.validation_utils import (
    handle_time_bounds,
    validate_and_normalize_time_range,
    validate_entity_exists,
    extract_effective_time_properties,
)
from app.infra.parameter import Parameter
from app.infra.quantity import Quantity
from app.infra.relation import EntityReference
from app.infra.relation import Relation
from app.infra.util import TimeSet
from app.infra.logging_setup import get_logger
from app.model.entity import Entity
from app.model.model import Model


log = get_logger(__name__)

class PulpConverter(Converter):
    """
    Converts a Model class object into a PuLP optimization problem.

    This converter handles the transformation of energy system models into linear programming
    problems that can be solved using the PuLP optimization library. It converts entities,
    quantities, and relations into PuLP variables and constraints while maintaining temporal
    relationships and handling time series data resampling.

    The converter supports:
    - Time-indexed variables for dynamic optimization
    - Linear constraints between variables
    - Temporal relationships (t-1, t+1 references)
    - Time series data resampling
    - Hierarchical entity structures

    Attributes
    ----------
    DEFAULT_TIME_RANGE_SIZE : int
        Default number of time steps when none is specified (10)

    Notes
    -----
    This converter is designed for linear programming problems only. Non-linear operations
    like variable multiplication or division by variables will raise ValueError.

    Examples
    --------
    >>> converter = PulpConverter()
    >>> model_vars = converter.convert_model(energy_model)
    >>> constraints = model_vars['battery.soc_constraint']
    """

    def __init__(self):
        super().__init__()
        self.__objects: Dict[str, Any] = {}
        self.__current_entity_id: Optional[str] = (
            None  # Track current entity being processed
        )

    def _register_converters(self):
        """
        PulpConverter doesn't need specialized entity converters.
        All entities use the default conversion logic.
        """
        # Intentionally empty - all entities use _convert_entity_default
        pass

    def _convert_entity_default(
        self,
        entity: Entity,
        time_set: Optional[TimeSet] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convert an Entity's quantities into pulp variables.
        
        Note: Sub-entity traversal is handled by the base Converter class.
        This method only converts the current entity's quantities.

        Parameters
        ----------
        entity : Entity
            The entity to convert (quantities only, not sub-entities).
        time_set : TimeSet, optional
            TimeSet object containing time configuration (number of steps, frequency, etc.).
            If None, uses the default time range size.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing pulp variables from this entity.
            Keys are in the format 'entity_id.quantity_name'.
        """
        log.debug(f"Using fallback default entity converter for '{entity.id}'")
        
        # Set current entity context for SelfReference resolution
        self.__current_entity_id = entity.id
        # Convert entity quantities (ONLY this entity, not sub-entities)
        entity_variables = {
            f"{entity.id}.{key}": self.convert_quantity(
                quantity,
                name=f"{entity.id}.{key}",
                time_set=time_set,
            )
            for key, quantity in entity.quantities.items()
        }
        log.debug(f"Converted entity '{entity.id}' quantities to variables: {list(entity_variables.keys())}")
        # Note: Sub-entities are handled by base class Converter.convert_entity()
        # Note: Relations are NOT converted here - they are handled in convert_model
        # after all entity variables are collected to avoid incomplete __objects

        return entity_variables

    def _convert_entity_relations(
        self,
        entity: Entity,
        time_set: Optional[TimeSet] = None,
    ) -> Dict[str, Any]:
        """
        Recursively convert relations for an entity and its sub-entities.

        This method should only be called after __objects has been fully populated
        with all entity variables.

        Parameters
        ----------
        entity : Entity
            The entity whose relations to convert
        time_set : TimeSet, optional
            TimeSet object containing time configuration

        Returns
        -------
        Dict[str, Any]
            Dictionary containing relation constraints
        """
        # Set current entity context for SelfReference resolution
        self.__current_entity_id = entity.id

        relation_constraints = {}

        # Convert relations to constraints
        for relation in entity.relations:
            constraints = self.convert_relation(relation, time_set=time_set)
            relation_constraints.update(constraints)

        # Recursively convert sub-entity relations
        for _, sub_entity in entity.sub_entities.items():
            relation_constraints.update(
                self._convert_entity_relations(sub_entity, time_set)
            )

        return relation_constraints

    def _prepare_conversion(
        self, model: Model, time_set: Optional[TimeSet], **kwargs
    ) -> tuple[TimeSet, Dict[str, Any]]:
        """
        Prepare PuLP conversion by extracting time_set and initializing state.
        
        Initializes __objects to empty dict for fresh conversion.
        """
        effective_time_set = extract_effective_time_properties(model, time_set)
        
        # Reset state for new conversion
        self.__objects = {}
        self.__current_entity_id = None
        
        context = {"skip_entities": kwargs.get("skip_entities", set())}
        return effective_time_set, context

    def _convert_entities(
        self,
        model: Model,
        time_set: TimeSet,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert all model entities to PuLP variables.
        
        Populates __objects with all entity variables for later relation conversion.
        """
        skip_entities = context.get("skip_entities", set())
        model_variables = {}
        
        for _, entity in model.entities.items():
            if type(entity) in skip_entities:
                continue
            model_variables.update(self.convert_entity(entity, time_set))
        
        # Store all entity variables in __objects for relation conversion
        self.__objects = dict(model_variables)
        
        return model_variables
    
    def _post_process_conversion(
        self,
        model: Model,
        result: Dict[str, Any],
        time_set: TimeSet,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Post-processing: Convert all relations to constraints.
        
        This is the second pass that requires all entity variables to be available.
        """
        skip_entities = context.get("skip_entities", set())
        
        # Convert relations for all entities
        for _, entity in model.entities.items():
            if type(entity) in skip_entities:
                continue
            relation_constraints = self._convert_entity_relations(entity, time_set)
            result.update(relation_constraints)
        
        return result

    def convert_quantity(
        self,
        quantity: Quantity,
        name: str,
        time_set: Optional[TimeSet] = None,
    ) -> Union[List[pulp.LpVariable], Any]:
        """
        Convert the time series data to a format suitable for pulp optimization.

        If the quantity is empty, create an empty pulp variable.
        If the quantity is a Parameter, return its value directly.
        Otherwise, return the values resampled to the specified time range and frequency.

        Parameters
        ----------
        quantity : Quantity
            The quantity to convert
        name : str
            The name for the generated variables
        time_set : TimeSet, optional
            TimeSet object containing time configuration. If None, uses default size.

        Returns
        -------
        Union[List[pulp.LpVariable], Any]
            Either a list of PuLP variables (for empty quantities) or the quantity values
        """
        if isinstance(quantity, Parameter):
            return quantity.value
        elif quantity.empty():
            normalized_time_range = validate_and_normalize_time_range(
                time_set, self.DEFAULT_TIME_RANGE_SIZE
            )
            return create_empty_pulp_var(name, len(normalized_time_range))
        else:
            # Extract time range and frequency from TimeSet
            time_range = time_set.number_of_time_steps if time_set else None
            freq = time_set.freq if time_set else None
            return quantity.value(time_set=time_range, freq=freq)

    def convert_relation(
        self,
        relation: Relation,
        entity_variables: Optional[Dict[str, Any]] = None,
        time_set: Optional[TimeSet] = None,
    ) -> Dict[str, List]:
        """
        Convert a Relation to PuLP constraints for each time step.

        This method dynamically assembles relations based on identified entity references
        and operations. For example:
        "battery1.discharge_power(t)<2*battery1.discharge_power(t-1)" becomes:
        for t in time_range:
            constraint = battery1.discharge_power[t] < 2 * battery1.discharge_power[t-1]

        Parameters
        ----------
        relation : Relation
            The relation to convert
        entity_variables : Dict[str, Any], optional
            Dictionary mapping entity IDs to their PuLP variables.
            If provided, updates __objects for backward compatibility.
            When called from convert_model, this is None since __objects is pre-populated.
        time_set : TimeSet, optional
            The TimeSet object containing full time information including time_points.
            Required for TimeConditionExpression conversions.

        Returns
        -------
        Dict[str, List]
            Dictionary containing the constraint name as key and list of constraints as value.

        Raises
        ------
        ValueError
            If any entity referenced in the relation is not found in __objects.

        Notes
        -----
        When called from convert_model(), __objects is pre-populated and entity_variables
        should be None. When called directly (e.g., in tests), entity_variables can be
        provided for backward compatibility.
        """
        # For backward compatibility: if entity_variables provided, update __objects
        if entity_variables is not None:
            self.__objects.update(entity_variables)

        # Validate that all required entities exist in __objects
        self._validate_relation_entities(relation)

        # Generate constraints for each time step
        constraints = self._generate_time_step_constraints(relation, time_set)

        return {relation.name: constraints}

    def _validate_relation_entities(self, relation: Relation) -> None:
        """
        Validate that all entities referenced in the relation exist.

        Parameters
        ----------
        relation : Relation
            The relation containing entity references to validate

        Raises
        ------
        ValueError
            If any entity referenced in the relation is not found in the objects dictionary.
        """
        for entity_id in relation.get_ids():
            # Skip validation for self-reference markers as they are resolved at conversion time
            if entity_id == "$":
                continue
            validate_entity_exists(entity_id, self.__objects)

    def _generate_time_step_constraints(
        self,
        relation: Relation,
        time_set,
    ) -> List:
        """
        Generate constraints for each time step in the range.

        Parameters
        ----------
        relation : Relation
            The relation to generate constraints for
        time_set : TimeSet or int, optional
            The full TimeSet object (required for TimeConditionExpression) or number of steps

        Returns
        -------
        List
            List of PuLP constraints, one for each time step where the constraint is not None
        """
        # Get number of time steps from TimeSet or use int directly
        if hasattr(time_set, "number_of_time_steps"):
            num_steps = time_set.number_of_time_steps
        elif isinstance(time_set, int):
            num_steps = time_set
        else:
            num_steps = self.DEFAULT_TIME_RANGE_SIZE

        constraints = []
        for t in range(num_steps):
            constraint = relation.expression.convert(self, t, time_set)
            if constraint is not None:
                constraints.append(constraint)
        return constraints

    def convert_entity_reference(
        self,
        entity_ref,
        entity_id: str,
        t: int,
        time_set: Optional[TimeSet] = None,
    ):
        """
        Convert an entity reference to a PuLP variable for the given time step.

        Parameters
        ----------
        entity_ref : EntityReference
            The entity reference object containing time offset information
        entity_id : str
            The entity ID to look up in the variables dictionary
        t : int
            Current time step
        time_set : TimeSet, optional
            TimeSet object (unused here but kept for consistency)

        Returns
        -------
        Union[pulp.LpVariable, Any]
            PuLP variable or expression for the specified time step

        Raises
        ------
        ValueError
            If the entity_id is not found in the objects dictionary.
        """
        actual_time = t + entity_ref.time_offset

        # Handle SelfReference-like entity IDs that start with "$."
        if entity_id.startswith("$."):
            return self.convert_self_reference(entity_ref, entity_id[2:], t, time_set)

        validate_entity_exists(entity_id, self.__objects)
        pulp_var = self.__objects[entity_id]

        return self._get_time_indexed_value(pulp_var, actual_time, entity_id)

    def _get_time_indexed_value(self, pulp_var, actual_time: int, entity_id: str):
        """
        Get the value from a time-indexed variable at the specified time step.

        Parameters
        ----------
        pulp_var : Any
            The PuLP variable or collection to access
        actual_time : int
            The time index to access
        entity_id : str
            The entity ID for error messages

        Returns
        -------
        Any
            The value at the specified time index, or the variable itself if not time-indexed
        """
        if isinstance(pulp_var, list):
            actual_time = handle_time_bounds(actual_time, pulp_var, entity_id)
            return pulp_var[actual_time]
        elif hasattr(pulp_var, "__getitem__"):  # Dict-like or other indexable
            return pulp_var[actual_time]
        else:
            # Single variable, not time-indexed
            return pulp_var

    def convert_binary_expression(
        self,
        binary_expr,
        left_result,
        right_result,
        operator,
        t: int,
        time_set: Optional[TimeSet] = None,
    ):
        """
        Convert a binary expression to PuLP constraint or arithmetic operation.

        Parameters
        ----------
        binary_expr : BinaryExpression
            The binary expression object (unused)
        left_result : Any
            Result of the left operand evaluation
        right_result : Any
            Result of the right operand evaluation
        operator : Operator
            The operator to apply between left and right operands
        t : int
            Current time step (unused)
        time_set : TimeSet, optional
            TimeSet object (unused)

        Returns
        -------
        Union[pulp.LpConstraint, pulp.LpExpression]
            PuLP constraint for comparison operators or expression for arithmetic operators

        Raises
        ------
        ValueError
            If the operator is not supported or violates linear programming constraints.
        """

        if is_comparison_operator(operator):
            return handle_comparison_operator(operator, left_result, right_result)
        elif is_arithmetic_operator(operator):
            return handle_arithmetic_operator(operator, left_result, right_result)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def convert_self_reference(
        self,
        self_ref,
        property_name: str,
        t: int,
        time_set: Optional[TimeSet] = None,
    ):
        """
        Convert a self reference to a PuLP variable for the given time step.
        Self references ($.property) are resolved to the current entity property.

        Parameters
        ----------
        self_ref : SelfReference
            The self reference object containing time offset information
        property_name : str
            The property name to resolve
        t : int
            Current time step
        time_set : TimeSet, optional
            TimeSet object (unused here but kept for consistency)

        Returns
        -------
        Union[pulp.LpVariable, Any]
            PuLP variable or expression for the specified time step

        Raises
        ------
        ValueError
            If no current entity context is available for resolution.
        """
        # Use current entity context to resolve self reference
        if self.__current_entity_id is None:
            raise ValueError(
                "No current entity context available for SelfReference resolution"
            )

        # Construct the full entity.property key
        entity_property_key = f"{self.__current_entity_id}.{property_name}"
        validate_entity_exists(entity_property_key, self.__objects)
        pulp_var = self.__objects[entity_property_key]

        # Calculate actual time with offset
        actual_time = t + self_ref.time_offset

        return self._get_time_indexed_value(pulp_var, actual_time, entity_property_key)

    def convert_literal(
        self,
        literal,
        value,
        t: int,
        time_set: Optional[TimeSet] = None,
    ):
        """
        Convert a literal value to its native representation.

        Parameters
        ----------
        literal : Literal
            The literal object (unused)
        value : Any
            The literal value
        t : int
            Current time step (unused)
        time_set : TimeSet, optional
            TimeSet object (unused)

        Returns
        -------
        Any
            The literal value as-is
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
        Convert a time condition expression to PuLP constraints based on time windows.

        For 'enabled' conditions, the entity variable is constrained to be non-zero
        within the time window and zero outside it. For 'disabled' conditions,
        the opposite applies.

        Parameters
        ----------
        time_condition_expression : TimeConditionExpression
            The time condition expression object
        entity : str
            The entity ID (e.g., 'heater1.p_in')
        condition : str
            The condition type ('enabled', 'disabled', etc.)
        between_time_index : array-like
            Index array of time steps that fall within the time window
        t : int
            Current time step
        """
        if condition == "enabled" and not between_time_index[t]:
            return (
                self.convert_entity_reference(EntityReference(entity, 0), entity, t)
                == 0
            )
        if condition == "disabled" and between_time_index[t]:
            return (
                self.convert_entity_reference(EntityReference(entity, 0), entity, t)
                == 0
            )
        return None


