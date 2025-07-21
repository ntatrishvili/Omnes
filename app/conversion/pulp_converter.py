from typing import override

import pulp

from app.conversion.converter import Converter
from app.infra.quantity import Quantity
from app.infra.relation import Relation
from app.model.entity import Entity
from app.model.model import Model


class PulpConverter(Converter):
    @override
    def convert_model(self, model: Model, time_set: int = None, new_freq: str = None):
        variables = {}
        for entity in model.entities:
            variables.update(entity.convert(time_set, new_freq, self))
        variables["time_set"] = range(time_set)
        return variables

    @override
    def convert_entity(self, entity: Entity, time_set: int = None, new_freq: str = None):
        """
        Convert an Entity and its sub-entities into a flat dictionary of pulp variables
        suitable for optimization.

        This method recursively traverses the entity hierarchy, resamples all time series
        data to the specified frequency, and converts each TimeseriesObject into pulp-compatible
        variables using its `convert` method.

        Parameters:
        ----------
        entity : Entity
            The root entity to convert (may have sub-entities).
        time_set : int, optional
            The number of time steps to represent in the pulp variables.
        new_freq : str, optional
            The target frequency to resample time series data to (e.g., '15min', '1H').

        Returns:
        -------
        dict
            A flat dictionary containing all pulp variables from the entity and its descendants.
        """
        variables = {
            f"{entity.id}.{key}": self.convert_quantity(self, quantity, name=f"{entity.id}.{key}", time_set=time_set,
                                                        freq=new_freq)
            for key, quantity in entity.quantities.items()
        }
        for sub_entity in entity.sub_entities:
            variables.update(self.convert_entity(sub_entity, time_set, new_freq))
        return variables

    @override
    def convert_quantity(self, quantity: Quantity, name: str, time_set: int = None, freq: str = None):
        """Convert the time series data to a format suitable for pulp optimization."""
        if quantity.empty():
            return create_empty_pulp_var(name, time_set)
        return quantity.get_values(time_set=time_set, freq=freq)

    @override
    def convert_relation(self, relation: Relation, time_set: int = None, new_freq: str = None):
        pass


def create_empty_pulp_var(name: str, time_set: int) -> list[pulp.LpVariable]:
    """
    Create a list of empty LpVariable with the specified name and time set
    """
    return [pulp.LpVariable(f"{name}_{t}", lowBound=0) for t in range(time_set)]
