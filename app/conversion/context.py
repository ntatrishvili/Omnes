from app.conversion.convert_optimization import convert
from app.conversion.converter import Converter
from app.infra.singleton import Singleton
from app.model.model import Model


class Context(Singleton):
    def __init__(self):
        super(Context, self).__init__()
        self.context = {}

    @property
    def context(self):
        return self.context

    @context.setter
    def context(self, context):
        self.context = context

    def __setitem__(self, key, value):
        self.context[key] = value

    def register_context(
        self, model: Model, converter: Converter, time_set: int, time_resolution: str
    ):
        self.context = convert(
            model,
            converter,
        )
