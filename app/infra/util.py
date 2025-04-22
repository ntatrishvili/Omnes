import pandas as pd
import os
import pulp

from app.model.timeseries_object import TimeseriesObject

def read_ts(filename: str, col: str) -> TimeseriesObject:
    """
    Read a csv file and return a timeseries object
    with the specified column and timestamp
    parameters:
    col: str: column name
    filename: str: input file name
    return: pd.DataFrame: DataFrame with the specified column and timestamp
    """
    input_path = get_input_path(filename)

    try:
        input_df = pd.read_csv(input_path, sep=";", header=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{input_path}' does not exist.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file '{input_path}' is empty or invalid.")
    
    if "timestamp" not in input_df.columns:
        raise KeyError(f"The column 'timestamp' is not found in the file '{filename}'.")

    input_df["timestamp"] = pd.to_datetime(input_df["timestamp"], format="%Y.%m.%d %H:%M")
    input_df.set_index("timestamp", inplace=True)

    if col not in input_df.columns:
        raise KeyError(f"The column '{col}' is not found in the file '{filename}'.{input_df}")
    
    result = input_df[col]
    
    return TimeseriesObject(result)


def get_input_path(filename: str = "input.csv") -> str:
    """
    Return the path to the input csv file.
    Returns data/input.csv by default #TODO add custom path
    """
    data_folder = os.path.join(os.path.dirname(__file__), "..", "..", "data")

    return os.path.join(data_folder, filename)


def create_empty_pulp_var(name: str, time_set: int) -> list[pulp.LpVariable]:
    """
    Create a list of empty LpVariable with the specified name and time set
    """
    return [pulp.LpVariable(f"P_{name}_{t}", lowBound=0) for t in range(time_set)]


def flatten(nested_list):
    """
    Flatten a nested list.
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)

    return flattened
