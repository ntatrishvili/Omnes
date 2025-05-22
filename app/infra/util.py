import os
import pulp


def get_input_path(filename: str = "input.csv") -> str:
    """
    Return the absolute path to the input CSV file.

    By default, returns the path to 'data/input.csv' relative to this file.

    :param filename: str: Name of the CSV file to locate.
    :raises FileNotFoundError: If the data folder or file does not exist.
    :return: Absolute path to the requested CSV file.
    """
    data_folder = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(
            f"The data folder '{data_folder}' does not exist. Please check the path."
        )

    filepath = os.path.join(data_folder, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"The file '{filename}' does not exist in the data folder '{data_folder}'."
        )

    return filepath


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
