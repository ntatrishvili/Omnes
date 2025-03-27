import json


# from app.operation.example_optimization import optimize
# from app.conversion.convert_optimization import convert
from app.model.model import Model

if __name__ == "__main__":
    # time_set, parameters = convert()
    # parameters += {"time_set": time_set}
    # optimize(parameters)
    with open("data/model_config.json", "r") as file:
        config = json.load(file)
    model = Model()
    model = model.build(config)
    print(model)
