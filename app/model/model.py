from typing import Optional

from app.conversion.converter import Converter
from app.conversion.pulp_converter import PulpConverter
from app.infra.util import get_input_path
from app.model.battery import Battery
from app.model.load import Load
from app.model.entity import Entity
from app.model.pv import PV
from app.model.slack import Slack


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
        self,
        id: Optional[str] = None,
        time_set: int = 0,
        frequency: str = "15min",
        **kwargs
    ):
        """
        Initialize the model with an optional name
        """
        self.id: str = id if id else "model"
        self.time_set: int = time_set
        self.frequency: str = frequency
        self.entities: list[Entity] = kwargs.get("entities", [])

    def add_entity(self, entity: Entity):
        self.entities.append(entity)

    @classmethod
    def build(cls, config: dict, time_set: int, frequency: str):
        """
        Build the model from a configuration dictionary
        """
        model = cls("model")
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

    def to_pulp(self, converter: Optional[Converter] = None, **kwargs):
        """
        Convert the model to pulp variables
        """
        time_set = kwargs.get("time_set", self.time_set)
        frequency = kwargs.get("frequency", self.frequency)
        converter = converter or PulpConverter()
        pulp_vars = {}
        for entity in self.entities:
            pulp_vars.update(entity.to_pulp(time_set, frequency, converter))
        pulp_vars["time_set"] = range(time_set)
        return pulp_vars

    def __str__(self):
        """
        String representation of the model
        """
        return "\n".join([str(entity) for entity in self.entities])
