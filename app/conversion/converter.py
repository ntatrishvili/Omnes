from app.infra.quantity import Quantity
from app.infra.relation import Relation
from app.model.entity import Entity
from app.model.model import Model


class Converter(object):
    def convert_entity(
        self, entity: Entity, time_set: int = None, new_freq: str = None
    ):
        raise NotImplementedError("Subclasses must implement 'convert'.")

    def convert_model(self, model: Model, time_set: int = None, new_freq: str = None):
        raise NotImplementedError("Subclasses must implement 'convert_model'.")

    def convert_quantity(
        self, quantity: Quantity, time_set: int = None, new_freq: str = None
    ):
        raise NotImplementedError("Subclasses must implement 'convert_quantity'.")

    def convert_relation(
        self, relation: Relation, time_set: int = None, new_freq: str = None
    ):
        raise NotImplementedError("Subclasses must implement 'convert_relation'.")
