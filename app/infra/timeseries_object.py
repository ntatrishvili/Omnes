import numpy as np
import pandas as pd

from app.infra.quantity import Quantity


class TimeseriesObject(Quantity):
    """
    A class representing a time series object.
    This class is used to handle time series data, including reading from CSV files,
    resampling, and normalizing frequencies.
    """

    def __init__(self, **kwargs):
        """
        Initialize the TimeseriesObject with either:
        - a pandas DataFrame (using the ``data`` keyword), or
        - a CSV file and column name (using the ``input_path`` and ``col`` keywords).

        If neither is provided, an empty DataFrame is used.

        :keyword data: The pandas DataFrame to initialize the object with.
        :type data: pandas.DataFrame, optional
        :keyword input_path: The path to the CSV file to read data from.
        :type input_path: str, optional
        :keyword col: The column name to extract from the CSV file.
        :type col: str, optional
        :keyword freq: str, optional
        """
        super().__init__(**kwargs)
        data = kwargs.pop("data", None)
        input_path = kwargs.pop("input_path", None)
        col = kwargs.pop("col", None)
        freq = kwargs.pop("freq", None)

        if isinstance(data, pd.DataFrame):
            self.data = pd.DataFrame(data)
        elif input_path is not None and col is not None:
            self.data = TimeseriesObject.read(input_path, col).data
        else:
            self.data = pd.DataFrame()

        if not self.data.empty:
            self.data.index = pd.to_datetime(self.data.index)

            if freq is not None:
                self.resample_to(freq)
            else:
                self.freq = self.normalize_freq(pd.infer_freq(self.data.index))
        else:
            self.freq = None

    @staticmethod
    def read(input_path: str, col: str) -> "TimeseriesObject":
        """
        Read a CSV file and return a TimeseriesObject with
        the specified column and timestamp parameters.

        :param input_path: path to the CSV file.
        :param col: str Name of the column to extract from the CSV file.
        :raises FileNotFoundError: If the specified file does not exist.
        :raises ValueError: If the file is empty or invalid.
        :raises KeyError: If the specified column is not found in the file.
        :return: TimeseriesObject A TimeseriesObject containing the specified column and timestamp as the index.
        """
        try:
            input_df = pd.read_csv(
                input_path,
                sep=";",
                header=0,
                index_col="timestamp",
                parse_dates=["timestamp"],
                date_format="%Y.%m.%d %H:%M",
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{input_path}' does not exist.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file '{input_path}' is empty or invalid.")

        if col not in input_df.columns:
            raise KeyError(
                f"The column '{col}' is not found in the file {input_path}'. {input_df}"
            )

        result = input_df[[col]]

        return TimeseriesObject(data=result)

    @classmethod
    def normalize_freq(cls, freq: str) -> str:
        """
        Normalize the frequency string to a standard format.
        :param freq: str
            The frequency string to normalize (e.g., 'H').
        :return: str
            The normalized frequency string (e.g., '1H').
        """
        if freq is not None and freq.isalpha():
            return f"1{freq}"
        return f"{freq}"

    def to_1h(self, closed="left") -> "TimeseriesObject":
        """
        Convert the time series to 1-hour frequency.
        :param closed:
            Which side of bin interval is closed. Default is 'left'.
        :raises ValueError:
            If the frequency is not set.
        :return: TimeseriesObject
            A TimeseriesObject resampled to 1-hour frequency.
        """
        if self.freq is None:
            raise ValueError("Frequency of the time series is not set.")
        return self.resample_to("1H", closed=closed)

    def to_15m(self, closed="left") -> "TimeseriesObject":
        """
        Convert the time series to 15-minute frequency.
        :param closed:
            Which side of bin interval is closed. Default is 'left'.
        :raises ValueError:
            If the frequency is not set.
        :return: TimeseriesObject
            A TimeseriesObject resampled to 15-minute frequency.
        """
        if self.freq is None:
            raise ValueError("Frequency of the time series is not set.")
        return self.resample_to("15min", closed=closed)

    def resample_to(
        self, new_freq, method=None, agg="mean", closed="right", in_place=False
    ) -> "TimeseriesObject":
        """
        Resample the stored time series to a new frequency.

        :param new_freq: str
            The new frequency to resample to (e.g., '15min', 'H').
        :param method: str, optional
            The resampling method to use ('interpolate', 'ffill', 'bfill', or 'agg'). Defaults to None.
        :param agg: str, optional
            The aggregation function to use if method='agg' (e.g., 'mean', 'sum', 'last'). Defaults to 'mean'.
        :param closed: str, optional
            Which side of bin interval is closed. ('right', 'left'). Defaults to 'right'.
        :param in_place: bool, optional
            Signals whether the object itself is modified at the end of the operation
        :raises ValueError:
            If the frequency cannot be inferred or if an unsupported method is used.
        :return: TimeseriesObject
            A TimeseriesObject resampled to the specified frequency.
        """
        if self.data.empty or self.freq == new_freq:
            return self

        try:
            print(f"Resampling from {self.freq} to {new_freq}")
            current_freq = TimeseriesObject.normalize_freq(
                pd.infer_freq(self.data.index)
            )
        except Exception as e:
            raise ValueError(f"Error inferring frequency: {e}")

        if current_freq is None:
            raise ValueError(
                "Cannot infer current frequency. Please specify method manually."
            )

        try:
            if method is None:
                if pd.Timedelta(new_freq) < pd.Timedelta(current_freq):
                    method = "interpolate"  # Upsampling
                else:
                    method = "agg"  # Downsampling

            if method == "interpolate":
                resampled = self.data.resample(new_freq, closed=closed).interpolate(
                    "linear"
                )
            elif method in ("ffill", "bfill"):
                resampled = getattr(
                    self.data.resample(new_freq, closed=closed), method
                )()
            elif method == "agg":
                resampled = self.data.resample(new_freq, closed=closed).agg(agg)
            else:
                raise ValueError(
                    "Unsupported method. Use 'interpolate', 'ffill', 'bfill', or 'agg'."
                )
        except Exception as e:
            raise ValueError(f"Error during resampling: {e}")

        if in_place:
            self.data = resampled
        return TimeseriesObject(data=resampled)

    def to_df(self) -> pd.DataFrame:
        """
        return: pd.DataFrame
            The original time series data.
        """
        return self.data

    def to_nd(self) -> np.ndarray:
        """
        return: pd.DataFrame
            The original time series data.
        """
        return self.data.values.flatten()

    def get_values(self, **kwargs):
        freq = kwargs.get("freq", self.freq)
        time_set = kwargs.get("time_set", len(self.data))
        if freq != self.freq:
            return self.resample_to(freq).to_nd()
        if time_set != len(self.data):
            return self.resample_to(freq).to_nd()[:time_set]
        return self.to_nd()

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying pandas DataFrame.
        Called only when attribute is not found in TimeseriesObject directly.
        """
        data_attr = getattr(self.data, name, None)
        if data_attr is not None:
            return data_attr
        raise AttributeError(f"'TimeseriesObject' object has no attribute '{name}'")

    def empty(self) -> bool:
        return self.data.empty

    def __eq__(self, other):
        return self.data == other.data and self.freq == other.freq
