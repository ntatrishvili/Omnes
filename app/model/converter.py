from app.model.entity import Entity


class Converter(Entity):
    pass


class WaterHeater(Converter):
    def __init__(self, **kwargs):
        kwargs.setdefault("input_vector", "electricity")
        kwargs.setdefault("output_vector", "heat")
        kwargs.setdefault("is_controlled", True)
        super().__init__(**kwargs)

    def __str__(self):
        return f"Water Heater '{self.id}' (controlled={self['is_controlled']})"
