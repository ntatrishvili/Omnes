import json
import configparser

from utils.logging_setup import init_logging, get_logger
from app.conversion.pulp_converter import PulpConverter
from app.operation.example_optimization import optimize
from app.model.model import Model

if __name__ == "__main__":
    init_logging(level="INFO", log_dir="logs", log_file="app.log")
    log = get_logger(__name__)
    log.info("Logging initialized")
    config = configparser.ConfigParser()
    config.read("config.ini")
    freq = config.get("time", "frequency")
    time_set = config.getint("time", "time_set")

    with open("data/model_config.json", "r") as file:
        config = json.load(file)
    model = Model.build(config, time_set, freq)
    log.info("Model built successfully")
    problem = PulpConverter().convert(model)
    log.info("Starting optimization")
    optimize(**problem)
    log.info("Optimization completed")
