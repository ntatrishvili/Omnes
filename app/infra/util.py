import os
import pulp
from pandas import date_range


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


class TimesetBuilder:
    @classmethod
    def create(cls, **kwargs):
        time_start = kwargs.get("time_start", None)
        time_end = kwargs.get("time_end", None)
        # TODO: Huge hack, how to handle?
        if time_start is None and time_end is None:
            time_start = "2019-01-01"
        number_of_time_steps = kwargs.get("number_of_time_steps", None)
        resolution = kwargs.get("resolution", None)
        dates = date_range(
            start=time_start,
            end=time_end,
            freq=resolution,
            periods=number_of_time_steps,
        )
        number_of_time_steps = dates.shape[0]
        resolution = dates.freq
        return TimeSet(time_start, time_end, resolution, number_of_time_steps, dates)


class TimeSet:
    def __init__(self, start, end, resolution, number_of_time_steps, time_points):
        self.start = start
        self.end = end
        self.resolution = resolution
        self.number_of_time_steps = number_of_time_steps
        self.time_points = time_points
