import json


from app.operation.example_optimization import optimize
from app.model.model import Model

if __name__ == "__main__":
    with open("data/model_config.json", "r") as file:
        config = json.load(file)
    model = Model.build(config, "./data/input.csv")
    problem = model.to_pulp()
    optimize(**problem)
