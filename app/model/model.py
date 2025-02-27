from app.model.unit import Unit


class Model:
    def __init__(self):
        self.units: list[Unit] = []

    def add_unit(self, unit: Unit):
        self.units.append(unit)
