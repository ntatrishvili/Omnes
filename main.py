import json
import configparser

from app.operation.example_optimization import optimize
from app.model.model import Model

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    time_set = config.getint("time", "time_set")
    with open("data/model_config.json", "r") as file:
        config = json.load(file)
    model = Model.build(config, time_set)
    problem = model.to_pulp()
    optimize(**problem)
