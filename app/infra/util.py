import pandas as pd
import os
import pulp


def fill_df(col: str) -> pd.DataFrame:
    """
    Read a csv file and return a DataFrame with the specified column and timpestamp
    parameters:
    col: str: column name
    return: pd.DataFrame: DataFrame with the specified column and timestamp
    """
    input_path = get_input_path()
    input_df = pd.read_csv(input_path, index_col=0, header=0)
    return input_df[col]


def get_input_path(filename: str = "input.csv") -> str:
    """
    Return the path to the input csv file.
    Returns data/input.csv by default #TODO add custom path
    """
    data_folder = os.path.join(os.path.dirname(__file__), "..", ".. ", "data")

    return os.path.join(data_folder, filename)


def create_empty_pulp_var(name: str, time_set: int) -> list[pulp.LpVariable]:
    """
    Create a list of empty LpVariable with the specified name and time set
    """
    return [pulp.LpVariable(f"P_{name}_{t}", lowBound=0) for t in range(time_set)]

def flatten(nested_list):
    """"
    Flatten a nested list.
    """
    flattened = []
    
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    
    return flattened
        