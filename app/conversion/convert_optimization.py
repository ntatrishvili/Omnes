from ..model.model import Model


def convert(model: Model) -> tuple[list, dict]:
    problem = model.to_pulp()

    return problem
