from app.operation.example_optimization import optimize
from app.conversion.convert_optimization import convert

if __name__ == "__main__":
    time_set, parameters = convert()
    parameters += {"time_set": time_set}
    optimize(parameters)
