from typing import Dict, Optional

from app.infra.quantity import Quantity, Parameter
from app.infra.timeseries_object_factory import (
    DefaultTimeseriesFactory,
    TimeseriesFactory,
)
from app.model.entity import Entity


class GenericEntity(Entity):
    """Represents a generic modelled object in the energy system.

    Use this class when an input object does not require a more specific
    subclass. GenericEntity stores named quantities (as Parameter objects),
    may contain nested sub-entities (inherited from Entity), and can hold
    relations or tags.

    Parameters
    ----------
    id : str | None
        Optional unique identifier for the entity. If omitted a random id may
        be supplied by the base Entity implementation.
    ts_factory : TimeseriesFactory, optional
        Factory used to create time-series objects for quantities. If not
        provided DefaultTimeseriesFactory() is used.
    **kwargs :
        Arbitrary named quantities to initialize on the entity. Each key/value
        pair is converted into a Parameter(value=...). Common examples are
        numeric parameters parsed from CSV input (e.g. 'nominal_power').

    Notes
    -----
    - Additional keyword arguments accepted by the base Entity class are
      forwarded (e.g. for relations, tags or sub-entities).
    - Quantities added via kwargs are wrapped as Parameter instances. If you
      need a different Quantity subclass create the entity and update
      ``entity.quantities`` manually.

    Attributes
    ----------
    quantities : Dict[str, Quantity]
        Mapping of names to Quantity/Parameter objects belonging to the entity.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs,
    ):
        """Initialize the GenericEntity.

        Keyword arguments supplied in ``kwargs`` are converted to Parameter
        objects and added to the entity's ``quantities`` mapping. Remaining
        behavior (id assignment, sub-entities, relations) is handled by the
        base Entity constructor.
        """
        super().__init__(id, ts_factory, **kwargs)
        self.quantities.update(
            {key: Parameter(value=value) for key, value in kwargs.items()}
        )

    def __str__(self):
        """Return a compact human-readable representation of the entity.

        Only Quantity instances contained in ``self.quantities`` are included
        in the string to avoid printing nested structures or non-quantity
        metadata.
        """
        parameters_string = ", ".join(
            [
                str(param)
                for _, param in self.quantities.items()
                if isinstance(param, Quantity)
            ]
        )
        return f"Generic entity '{self.id}' containing: [{parameters_string}]"
