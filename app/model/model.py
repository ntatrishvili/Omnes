import secrets
from typing import Optional

from utils.logging_setup import get_logger
from app.infra.util import get_input_path, TimesetBuilder, TimeSet
from app.model.generator.pv import PV
from app.model.entity import Entity
from app.model.generator.pv import PV
from app.model.load.load import Load
from app.model.slack import Slack
from app.model.storage.battery import Battery

log = get_logger(__name__)


class Model:
    """
    Represents the entire simulation or optimization model.

    The Model serves as the container for all top-level entities, configuration parameters,
    time settings, and solver-specific logic. It provides an interface to convert the entire
    system into an optimization problem.

    Attributes:
        - id (Optional[str]): Identifies of the model.
        - entities (list[Entity]): Top-level entities (e.g., buses, generators, loads).
        - time_set (int): Number of time steps in the simulation.
        - frequency (str): Time resolution (e.g., '1h', '15min').
    """

    def __init__(
        self, id: Optional[str] = None, timeset_builder=TimesetBuilder(), **kwargs
    ):
        """
        Initialize the model with an optional name
        """
        self.id: str = id if id is not None else secrets.token_hex(16)
        self.time_set: TimeSet = timeset_builder.create(
            kwargs.pop("time_kwargs", {}), **kwargs
        )
        ent_list = kwargs.pop("entities", [])
        self.entities: dict[str, Entity] = {e.id: e for e in ent_list}

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity

    def __getitem__(self, id):
        if id in self.entities:
            return self.entities[id]
        else:
            for _, entity in self.entities.items():
                if id not in entity:
                    continue
                return entity[id]
        raise KeyError(f"Entity with id '{id}' not found in model '{self.id}'")

    def set(self, items_to_set):
        for item_id, item in items_to_set.items():
            if "." in item_id:
                entity_id, quantity_id = item_id.split(".")
                self[entity_id].quantities[quantity_id].set_value(
                    item, time_set=self.time_set
                )
            else:
                self.add_entity(item)

    @property
    def number_of_time_steps(self):
        return self.time_set.number_of_time_steps

    @property
    def frequency(self):
        return self.time_set.resolution

    @classmethod
    def build(cls, id, config: dict, time_set: int, frequency: str):
        """
        Build the model from a configuration dictionary
        """
        log.info(f"Building model '{id}' with {time_set} time steps")
        model = cls(id, number_of_time_steps=time_set, resolution=frequency)
        for entity_name, content in config.items():
            entity = Entity(entity_name)
            for pv_id, info in content["pvs"].items():
                pv = PV(
                    id=pv_id,
                    input_path=get_input_path(info["filename"]),
                    col=pv_id,
                    frequency=model.frequency,
                )
                entity.add_sub_entity(pv)
            for cs_id, info in content["consumers"].items():
                cs = Load(
                    id=cs_id,
                    input_path=get_input_path(info["filename"]),
                    col=cs_id,
                    frequency=model.frequency,
                )
                entity.add_sub_entity(cs)
            for b_id, info in content["batteries"].items():
                b = Battery(
                    b_id,
                    max_charge_rate=info["nominal_power"],
                    max_discharge_rate=info["nominal_power"],
                    capacity=info["capacity"],
                )
                entity.add_sub_entity(b)
            model.add_entity(entity)
        model.add_entity(Slack(id="slack"))
        log.info(f"Model built with {len(model.entities)} entities")
        return model

    def convert(self, converter, time_set: int = None, new_freq: str = None):
        """
        Convert the model to an optimization/simulation problem
        """
        log.info(f"Converting model with {time_set} steps and {new_freq} freq.")
        return converter.convert_model(self, time_set=time_set, new_freq=new_freq)

    def __str__(self):
        """
        String representation of the model
        """
        return "\n".join([str(entity) for _, entity in self.entities.items()])

    @frequency.setter
    def frequency(self, value):
        self.time_set.resolution = value
