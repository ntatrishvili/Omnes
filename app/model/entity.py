import secrets
from collections import defaultdict
from typing import Optional, Dict

from app.conversion.converter import Converter
from app.conversion.pulp_converter import PulpConverter
from app.model.timeseries_object import TimeseriesObject
from app.model.timeseries_object_factory import (
    TimeseriesFactory,
    DefaultTimeseriesFactory,
)


class Entity:
    def __init__(
        self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs
    ):
        """
        Initialize the entity with an optional id.
        """
        self.id = str(id) if id is not None else secrets.token_hex(16)
        self.quantities: Dict[str, TimeseriesObject] = {}
        self.parameters = defaultdict(float)
        self.sub_entities: list[Entity] = []
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
        if item in self.parameters:
            return self.parameters[item]
        elif item in self.quantities:
            return self.quantities[item]
        else:
            raise KeyError(f"'{item}' not found in parameters or quantities")
