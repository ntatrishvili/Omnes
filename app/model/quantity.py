from abc import abstractmethod

from app.infra.util import create_empty_pulp_var


class Quantity:
    def __init__(self, **kwargs): ...

    @abstractmethod
    def to_pulp(self, name: str, freq: str, time_set: int): ...

    @abstractmethod
    def __eq__(self, other): ...


class Parameter(Quantity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = kwargs.get("value", None)

    def to_pulp(self, name: str, freq: str, time_set: int):
        if self.value is None:
            return create_empty_pulp_var(name, time_set)
        return self.value

    def __str__(self):
        return f"{self.value}"

    def __eq__(self, other):
        try:
            return float(self.value) == float(other)
        except (TypeError, ValueError):
            return False
