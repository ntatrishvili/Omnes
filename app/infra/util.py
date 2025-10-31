import os

from pandas import date_range

from utils.logging_setup import get_logger

log = get_logger(__name__)


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


def try_convert(value, sample):
    import numbers
    import json

    # If no sample type to infer, accept the value as-is
    if sample is None:
        return value

    target_type = type(sample)

    # already correct type
    if isinstance(value, target_type):
        return value

    # Numeric types (int, float, np.*)
    if issubclass(target_type, numbers.Number):
        try:
            return target_type(value)
        except Exception:
            # try cleaning common numeric string formats
            if isinstance(value, str):
                cleaned = value.replace(",", "").strip()
                return target_type(cleaned)
            raise

    # Boolean handling from common strings
    if target_type is bool:
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("true", "1", "yes", "y", "t"):
                return True
            if s in ("false", "0", "no", "n", "f"):
                return False
        return bool(value)

    # Sequence / mapping conversions
    if target_type in (list, tuple, dict, str):
        try:
            return target_type(value)
        except Exception:
            # try JSON decode for strings to list/dict
            if isinstance(value, str) and target_type in (list, dict):
                try:
                    parsed = json.loads(value)
                    # if parsed already matching expected structure, return converted
                    if isinstance(parsed, (list, dict)):
                        return target_type(parsed)
                except Exception:
                    pass
            raise

    # Fallback: try to call the type on the value
    try:
        return target_type(value)
    except Exception:
        raise


class TimesetBuilder:
    @classmethod
    def create(cls, time_kwargs=None, **kwargs):
        if time_kwargs is None:
            time_kwargs = {}
        time_start = kwargs.pop("time_start", None)
        time_end = kwargs.pop("time_end", None)
        tz = kwargs.pop("tz", None)
        # TODO: Huge hack, how to handle?
        if time_start is None and time_end is None:
            time_start = "1970-01-01"
        number_of_time_steps = kwargs.pop("number_of_time_steps", None)
        resolution = kwargs.pop("resolution", None)
        dates = date_range(
            start=time_start,
            end=time_end,
            freq=resolution,
            periods=number_of_time_steps,
            **time_kwargs,
        )
        number_of_time_steps = dates.shape[0]
        log.info(f"Created timeset with {number_of_time_steps} time steps")
        resolution = dates.freq
        return TimeSet(
            time_start, time_end, resolution, number_of_time_steps, dates, tz
        )


class TimeSet:
    def __init__(
        self, start, end, resolution, number_of_time_steps, time_points, tz=None
    ):
        self.start = start
        self.end = end
        self.resolution = resolution
        self.number_of_time_steps = number_of_time_steps
        self.time_points = time_points
        self.tz = tz
