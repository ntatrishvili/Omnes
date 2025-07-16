import secrets
from typing import Optional

from pandas import date_range

from app.conversion.converter import Converter
from app.conversion.pulp_converter import PulpConverter
from app.infra.util import get_input_path
from app.model.generator.pv import PV
from app.model.entity import Entity
from app.model.load.load import Load
from app.model.slack import Slack
from app.model.storage.battery import Battery


class TimesetBuilder:
    @classmethod
    def create(cls, **kwargs):
        time_start = kwargs.get("time_start", None)
        time_end = kwargs.get("time_end", None)
        # TODO: Huge hack, how to handle?
        if time_start is None and time_end is None:
            time_start = "2019-01-01"
        number_of_time_steps = kwargs.get("number_of_time_steps", None)
        resolution = kwargs.get("resolution", None)
        dates = date_range(
            start=time_start,
            end=time_end,
            freq=resolution,
            periods=number_of_time_steps,
        )
        number_of_time_steps = dates.shape[0]
        resolution = dates.freq
        return TimeSet(time_start, time_end, resolution, number_of_time_steps, dates)


class TimeSet:
    def __init__(self, start, end, resolution, number_of_time_steps, time_points):
        self.start = start
        self.end = end
        self.resolution = resolution
        self.number_of_time_steps = number_of_time_steps
        self.time_points = time_points


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
        self.time_set: TimeSet = timeset_builder.create(**kwargs)
        self.entities: list[Entity] = kwargs.get("entities", [])

    def add_entity(self, entity: Entity):
        self.entities.append(entity)

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
        model = cls(id, number_of_time_steps=time_set, resolution=frequency)
        model.time_set = time_set
        model.frequency = frequency
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
                    b_id, max_power=info["nominal_power"], capacity=info["capacity"]
                )
                entity.add_sub_entity(b)
            model.add_entity(entity)
        model.add_entity(Slack(id="slack"))
        return model

    def convert(self, converter: Optional[Converter] = None, **kwargs):
        """
        Convert the model to an optimization/simulation problem
        """
        number_of_time_steps = kwargs.get("time_set", self.number_of_time_steps)
        frequency = kwargs.get("frequency", self.frequency)
        variables = {}
        for entity in self.entities:
            variables.update(entity.convert(number_of_time_steps, frequency, converter))
        variables["time_set"] = range(number_of_time_steps)
        return variables

    def __str__(self):
        """
        String representation of the model
        """
        return "\n".join([str(entity) for entity in self.entities])

    @frequency.setter
    def frequency(self, value):
        self.time_set.resolution = value
