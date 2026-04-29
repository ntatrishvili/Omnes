"""Read-only projections of models for specific optimization runs."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.model.entity import Entity
    from app.model.model import Model
    from app.infra.timeseries_object import TimeseriesObject


class QuantityRunView:
    """Wraps TimeSeriesObject, returns run-specific values.
    
    Provides access to optimization results with fallback chain:
    1. Optimization result (if available)
    2. Resampled/aligned data (if available)
    3. Raw data
    """
    
    def __init__(self, quantity: "TimeseriesObject", run_id: str):
        self._quantity = quantity
        self._run_id = run_id
    
    @property
    def value(self):
        """Return result → aligned → raw (in priority order).
        
        :returns: Optimization result, aligned data, or raw data
        """
        run = self._quantity.run(self._run_id)
        if run.result is not None:
            return run.result
        if run.aligned is not None:
            return run.aligned
        return self._quantity.value()  # Raw data fallback
    
    def __getattr__(self, name: str):
        """Proxy other attributes to underlying quantity."""
        return getattr(self._quantity, name)


class EntityRunView:
    """Wraps Entity, proxies quantities with run awareness.
    
    Provides access to entity quantities through RunView when they are
    TimeSeriesObjects, enabling transparent access to run-specific data.
    """
    
    def __init__(self, entity: "Entity", run_id: str):
        self.entity = entity
        self.run_id = run_id
        self.id = entity.id  # Pass through
        self.sub_entities = {
            e_id: EntityRunView(sub_e, run_id)
            for e_id, sub_e in entity.sub_entities.items()
        }
    
    def __getattr__(self, name: str):
        """Proxy quantity/entity access with run awareness.
        
        :param str name: Attribute name (quantity or sub-entity ID)
        :returns: QuantityRunView for TimeSeriesObjects, raw quantity otherwise
        :raises AttributeError: If name not found
        """
        if name in self.entity.quantities:
            qty = self.entity.quantities[name]
            # Import here to avoid circular dependency
            from app.infra.timeseries_object import TimeseriesObject
            if isinstance(qty, TimeseriesObject):
                return QuantityRunView(qty, self.run_id)
            return qty
        if name in self.entity.sub_entities:
            return self.sub_entities[name]
        raise AttributeError(f"Quantity/Entity '{name}' not found in entity '{self.id}'")


class RunView:
    """Read-only projection of model for a specific run.
    
    Provides clean access to model data for a specific optimization run,
    with automatic fallback to optimization results, aligned data, or raw data.
    """
    
    def __init__(self, model: "Model", run_id: str):
        self.model = model
        self.run_id = run_id
        self.id = model.id  # Pass through
        self.time_set = model.time_set #TODO: get correct timeset for run
        self.entities = {
            e_id: EntityRunView(entity, run_id)
            for e_id, entity in model.entities.items()
        }
    
    def __getitem__(self, entity_id: str) -> EntityRunView:
        """Access entity by ID.
        
        :param str entity_id: Entity ID
        :returns EntityRunView: Wrapped entity with run awareness
        :raises KeyError: If entity not found
        """
        return self.entities[entity_id]
    
    def __getattr__(self, name: str) -> EntityRunView:
        """Access entity by attribute name.
        
        :param str name: Entity ID (used as attribute)
        :returns EntityRunView: Wrapped entity with run awareness
        :raises AttributeError: If entity not found
        """
        if name in self.entities:
            return self.entities[name]
        raise AttributeError(f"Entity '{name}' not in model")
