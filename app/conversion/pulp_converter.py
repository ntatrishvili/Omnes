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
    validate_and_normalize_time_set,
    validate_entity_exists,
    extract_effective_time_properties,
)
from app.infra.quantity import Parameter, Quantity
from app.infra.relation import Relation
from app.model.entity import Entity
from app.model.model import Model


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
    DEFAULT_TIME_SET_SIZE : int
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
        time_set: Optional[Union[int, range]] = None,
        new_freq: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convert an Entity and its sub-entities into a flat dictionary of pulp variables.

        This method recursively traverses the entity hierarchy, resamples all time series
        data to the specified frequency, and converts each TimeseriesObject into pulp-compatible
        variables using its `convert` method.

        Parameters
        ----------
        entity : Entity
            The root entity to convert (may have sub-entities).
        time_set : Optional[Union[int, range]], optional
            The number of time steps to represent in the pulp variables.
            If None, uses the default time set size.
        new_freq : Optional[str], optional
            The target frequency to resample time series data to (e.g., '15min', '1H').

        Returns
        -------
        Dict[str, Any]
            A flat dictionary containing all pulp variables from the entity and its descendants.
            Keys are in the format 'entity_id.quantity_name' for quantities and 'relation_name'
            for constraints.
        """
        # Convert entity quantities
        entity_variables = {
            f"{entity.id}.{key}": self.convert_quantity(
                quantity,
                name=f"{entity.id}.{key}",
                time_set=time_set,
                freq=new_freq,
            )
            for key, quantity in entity.quantities.items()
        }

        # Recursively convert sub-entities
        for sub_entity in entity.sub_entities:
            entity_variables.update(self.convert_entity(sub_entity, time_set, new_freq))

        # Convert relations to constraints
        for relation in entity.relations:
            variable_mapping = {
                id: entity_variables.get(id) for id in relation.expression.get_ids()
            }
            relation_constraints = self.convert_relation(
                relation, variable_mapping, time_set=time_set, new_freq=new_freq
            )
            entity_variables.update(relation_constraints)

        return entity_variables

    def convert_model(
        self,
        model: Model,
        time_set: Optional[Union[int, range]] = None,
        new_freq: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convert the model to an optimization/simulation problem.

        Converts the model's entities and their quantities into a flat dictionary
        of pulp variables suitable for optimization. It also handles the resampling of time series
        data to the specified frequency and time set.

        Parameters
        ----------
        model : Model
            The model to convert.
        time_set : Optional[Union[int, range]], optional
            The number of time steps to represent in the pulp variables.
            If None, uses model.number_of_time_steps.
        new_freq : Optional[str], optional
            The target frequency to resample time series data to (e.g., '15min', '1H').
            If None, uses model.frequency.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all pulp variables and time set information.
            Includes a 'time_set' key with the range of time steps.
        """
        # Use model defaults if not specified
        effective_freq, effective_time_set = extract_effective_time_properties(
            model, new_freq, time_set
        )

        # Convert all entities
        model_variables = {}
        for entity in model.entities:
            model_variables.update(
                entity.convert(effective_time_set, effective_freq, self)
            )

        # Add time set information
        time_range = validate_and_normalize_time_set(
            effective_time_set, self.DEFAULT_TIME_SET_SIZE
        )
        model_variables["time_set"] = time_range

        return model_variables

    def convert_quantity(
        self,
        quantity: Quantity,
        name: str,
        time_set: Optional[Union[int, range]] = None,
        freq: Optional[str] = None,
    ) -> Union[List[pulp.LpVariable], Any]:
        """
        Convert the time series data to a format suitable for pulp optimization.

        If the quantity is empty, create an empty pulp variable.
        If the quantity is a Parameter, return its value directly.
        Otherwise, return the values resampled to the specified time set and frequency.

        Parameters
        ----------
        quantity : Quantity
            The quantity to convert
        name : str
            The name for the generated variables
        time_set : Optional[Union[int, range]], optional
            The time set specification. If None, uses default size.
        freq : Optional[str], optional
            The frequency for resampling

        Returns
        -------
        Union[List[pulp.LpVariable], Any]
            Either a list of PuLP variables (for empty quantities) or the quantity values
        """
        if isinstance(quantity, Parameter):
            return quantity.value
        elif quantity.empty():
            normalized_time_set = validate_and_normalize_time_set(
                time_set, self.DEFAULT_TIME_SET_SIZE
            )
            return create_empty_pulp_var(name, len(normalized_time_set))
        else:
            return quantity.value(time_set=time_set, freq=freq)

    def convert_relation(
        self,
        relation: Relation,
        entity_variables: Dict[str, Any],
        time_set: Optional[Union[int, range]] = None,
        new_freq: Optional[str] = None,
    ) -> Dict[str, List]:
        """
        Convert a Relation to PuLP constraints for each time step.

        This method dynamically assembles relations based on identified entity references
        and operations. For example:
        "battery1.discharge_power(t)<2*battery1.discharge_power(t-1)" becomes:
        for t in time_set:
            constraint = battery1.discharge_power[t] < 2 * battery1.discharge_power[t-1]

        Parameters
        ----------
        relation : Relation
            The relation to convert
        entity_variables : Dict[str, Any]
            Dictionary mapping entity IDs to their PuLP variables
        time_set : Optional[Union[int, range]], optional
            The time set to iterate over. If None, uses default size.
        new_freq : Optional[str], optional
            Frequency (kept for consistency with interface)

        Returns
        -------
        Dict[str, List]
            Dictionary containing the constraint name as key and list of constraints as value.

        Raises
        ------
        ValueError
            If any entity referenced in the relation is not found in entity_variables.
        """
        time_range = validate_and_normalize_time_set(
            time_set, self.DEFAULT_TIME_SET_SIZE
        )

        # Store entity variables for use by convert methods
        self.__objects = entity_variables

        # Validate that all required entities exist
        self._validate_relation_entities(relation)

        # Generate constraints for each time step
        constraints = self._generate_time_step_constraints(
            relation, time_range, new_freq
        )

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
            validate_entity_exists(entity_id, self.__objects)

    def _generate_time_step_constraints(
        self, relation: Relation, time_range: range, new_freq: Optional[str]
    ) -> List:
        """
        Generate constraints for each time step in the range.

        Parameters
        ----------
        relation : Relation
            The relation to generate constraints for
        time_range : range
            The range of time steps to iterate over
        new_freq : Optional[str]
            Target frequency for time series data

        Returns
        -------
        List
            List of PuLP constraints, one for each time step where the constraint is not None
        """
        constraints = []
        for t in time_range:
            constraint = relation.expression.convert(self, t, len(time_range), new_freq)
            if constraint is not None:
                constraints.append(constraint)
        return constraints

    def convert_entity_reference(
        self,
        entity_ref,
        entity_id: str,
        t: int,
        time_set: Optional[Union[int, range]] = None,
        new_freq: Optional[str] = None,
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
        time_set : Optional[Union[int, range]], optional
            Total time set (unused here but kept for consistency)
        new_freq : Optional[str], optional
            Frequency (unused here but kept for consistency)

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

        validate_entity_exists(entity_id, self.__objects)
        pulp_var = self.__objects[entity_id]

        # Handle time-indexed variables
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
        time_set: Optional[Union[int, range]] = None,
        new_freq: Optional[str] = None,
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
        time_set : Optional[Union[int, range]], optional
            Time set (unused)
        new_freq : Optional[str], optional
            Frequency (unused)

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
