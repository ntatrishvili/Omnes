import json
import numbers
import os

import pandas as pd
from pandas import date_range

from app.infra.logging_setup import get_logger
import hashlib

log = get_logger(__name__)


def get_input_path(filename: str = "input.csv") -> str:
    """
    Return the absolute path to the input CSV file.

    By default, returns the path to 'config/input.csv' relative to this file.

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


def cast_like(value, sample):
    # If no sample type to infer, accept the value as-is
    if sample is None:
        return value

    target_type = type(sample)

    # already correct type
    if isinstance(value, target_type):
        return value

    # Only perform conversions for numeric target types (exclude bool)
    if target_type is not bool and issubclass(target_type, numbers.Number):
        try:
            return target_type(value)
        except Exception:
            # try cleaning common numeric string formats
            if isinstance(value, str):
                cleaned = value.replace(",", "").strip()
                return target_type(cleaned)
            raise

    # For any non-numeric target types (including bool), do not attempt conversion; return as-is
    return value


class TimesetBuilder:
    @classmethod
    def create(cls, time_kwargs=None, **kwargs):
        """
        Create a TimeSet using pandas date_range.

        pandas.date_range requires exactly three of: start, end, periods, freq.
        This method intelligently selects which parameters to use based on what's provided.

        Parameters
        ----------
        time_kwargs : dict, optional
            Additional kwargs to pass to pandas.date_range
        time_start : str, optional
            Start time/date
        time_end : str, optional
            End time/date
        resolution : str, optional
            Frequency string (e.g., '1h', '15min')
        number_of_time_steps : int, optional
            Number of time steps (periods)
        tz : str, optional
            Timezone

        Returns
        -------
        TimeSet
            A properly configured TimeSet object
        """
        if time_kwargs is None:
            time_kwargs = {}
        time_start = kwargs.pop("time_start", None)
        time_end = kwargs.pop("time_end", None)
        tz = kwargs.pop("tz", None)
        number_of_time_steps = kwargs.pop("number_of_time_steps", None)
        resolution = kwargs.pop("resolution", None)

        # Default time_start if neither start nor end is provided
        if time_start is None and time_end is None:
            time_start = "1970-01-01"

        # pandas.date_range requires exactly 3 of: start, end, periods, freq
        # Count how many we have
        params = {
            "start": time_start,
            "end": time_end,
            "periods": number_of_time_steps,
            "freq": resolution,
        }

        non_none_count = sum(1 for v in params.values() if v is not None)

        # If we have all 4 parameters, we need to drop one
        # Priority: prefer start + periods + freq (drop end)
        if non_none_count == 4:
            time_end = None
        # If we have fewer than 3, we might need defaults
        elif non_none_count < 3:
            # If we have start but no freq and no end, default freq
            if time_start is not None and resolution is None and time_end is None:
                resolution = "h"  # Default to hourly
            # If we have no periods and have start + freq, default periods
            if (
                number_of_time_steps is None
                and time_start is not None
                and resolution is not None
            ):
                number_of_time_steps = 10  # Default number of steps

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
    """
    Represents a time configuration for model conversion and simulation.

    Attributes
    ----------
    start : str
        Start time/date of the time set
    end : str
        End time/date of the time set
    resolution : str
        Time resolution/frequency (e.g., '1H', '15min')
    number_of_time_steps : int
        Total number of time steps
    time_points : pandas.DatetimeIndex
        The actual time points
    tz : str, optional
        Timezone information
    """

    def __init__(
        self, start, end, resolution, number_of_time_steps, time_points, tz=None
    ):
        self.start = start
        self.end = end
        self.resolution = resolution
        self.number_of_time_steps = number_of_time_steps
        self.time_points = time_points
        self.tz = tz

    @property
    def freq(self) -> str:
        """
        Alias for resolution - frequency of the time set.

        Returns the frequency as a properly formatted pandas offset string
        (e.g., '1h', '15min', '1D').

        :return: The frequency/resolution string in pandas offset format
        """
        if self.resolution is None:
            return ""

        # If resolution is already a pandas offset, use its freqstr
        if hasattr(self.resolution, "freqstr"):
            offset = self.resolution.freqstr

        else:
            try:
                from pandas import tseries

                offset = tseries.frequencies.to_offset(self.resolution).freqstr
            except (ValueError, TypeError, AttributeError):
                offset = str(self.resolution)

        if not offset[0].isdigit():
            offset = f"1{offset}"

        return offset

    @property
    def hex_id(self) -> str:
        """
        Generate a unique hex identifier based on TimeSet properties.

        :return: 8-character hex string uniquely identifying this TimeSet configuration.
        """
        identity_string = (
            f"{self.start}|{self.end}|{self.resolution}|"
            f"{self.number_of_time_steps}|{self.tz}"
        )
        hash_digest = hashlib.md5(identity_string.encode()).hexdigest()
        return hash_digest[:8]

    def __repr__(self):
        return (
            f"TimeSet(start={self.start}, end={self.end}, "
            f"resolution={self.resolution}, steps={self.number_of_time_steps})"
        )
