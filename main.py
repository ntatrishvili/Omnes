import configparser
import json

from app.conversion.pulp_converter import PulpConverter
from app.model.model import Model
from app.operation.example_optimization import optimize_energy_system

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    freq = config.get("time", "frequency")
    time_set = config.getint("time", "time_set")

    with open("data/model_config.json", "r") as file:
        config = json.load(file)
    model = Model.build("model", config, time_set, frequency=freq)
    problem = PulpConverter().convert_model(model)
    optimize_energy_system(**problem)
