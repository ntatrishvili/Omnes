from typing import Optional

from app.infra.util import flatten, get_input_path
from app.model.battery import Battery
from app.model.consumer import Consumer
from app.model.entity import Entity
from app.model.pv import PV
from app.model.slack import Slack


class Model:
    def __init__(self, id: Optional[str] = None, time_set: int = 0, frequency: str = "15min", ):
        """
        Initialize the model with an optional name
        """
        self.id: str = id if id else "model"
        self.time_set: int = time_set
        self.frequency: str = frequency
        self.entities: list[Entity] = []

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
                pv = PV(id=pv_id, input_path=get_input_path(info["filename"]), col=pv_id, frequency=model.frequency)
                entity.add_sub_entity(pv)
            for cs_id, info in content["consumers"].items():
                cs = Consumer(id=cs_id, input_path=get_input_path(info["filename"]), col=cs_id,
                              frequency=model.frequency)
                entity.add_sub_entity(cs)
            for b_id, info in content["batteries"].items():
                b = Battery(b_id, max_power=info["nominal_power"], capacity=info["capacity"])
                entity.add_sub_entity(b)
            model.add_entity(entity)
        model.add_entity(Slack(id="slack"))
        return model

    def to_pulp(self):
        """
        Convert the model to pulp variables
        """
        pulp_vars = []
        for entity in self.entities:
            pulp_vars.extend(entity.to_pulp(self.time_set, self.frequency))
        pulp_vars = flatten(pulp_vars)
        pulp_vars = {k: v for d in pulp_vars for k, v in d.items()}
        pulp_vars["time_set"] = range(self.time_set)
        return pulp_vars

    def __str__(self):
        """
        String representation of the model
        """
        return "\n".join([str(entity) for entity in self.entities])
