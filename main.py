import json

from app.infra.configuration import Config
from app.infra.logging_setup import init_logging, get_logger
from app.conversion.pulp_converter import PulpConverter
from app.model.model import Model

from app.operation.example_optimization import optimize_energy_system


def read_model():
    config = Config()
    freq = config.get("time", "frequency")
    time_set = config.getint("time", "time_set")

    with open("config/model_config.json", "r") as file:
        config = json.load(file)
    return Model.build("model", config, time_set, freq)


if __name__ == "__main__":
    init_logging(
        level="DEBUG",
        log_dir="logs",
        log_file="app.log",
    )
    log = get_logger(__name__)
    log.info("Logging initialized")
    model = read_model()
    log.info("Model built successfully")
    problem = PulpConverter().convert_model(model)
    log.info("Starting optimization")
    optimize_energy_system(**problem)
    log.info("Optimization completed")
