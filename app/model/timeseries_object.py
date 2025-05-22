import datetime
import pulp
import pandas as pd


class TimeseriesObject:
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
        """
        data = kwargs.get("data", None)
        input_path = kwargs.get("input_path", None)
        col = kwargs.get("col", None)

        if isinstance(data, pd.DataFrame):
            self.data = pd.DataFrame(data)
        elif input_path is not None and col is not None:
            self.data = TimeseriesObject.read(input_path, col).data
        else:
            self.data = pd.DataFrame()

        if not self.data.empty:
            self.data.index = pd.to_datetime(self.data.index)
            inferred = pd.infer_freq(self.data.index)
            self.freq: datetime.timedelta = self.normalize_freq(inferred)
        else:
            self.freq: datetime.timedelta = None

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
            return "1" + freq
        return freq

    def to_1h(self) -> "TimeseriesObject":
        """
        Convert the time series to 1-hour frequency.

        :raises ValueError:
            If the frequency is not set.
        :return: TimeseriesObject
            A TimeseriesObject resampled to 1-hour frequency.
        """
        if self.freq is None:
            raise ValueError("Frequency of the time series is not set.")
        return self.resample_to("1H")

    def to_15m(self) -> "TimeseriesObject":
        """
        Convert the time series to 15-minute frequency.

        :raises ValueError:
            If the frequency is not set.
        :return: TimeseriesObject
            A TimeseriesObject resampled to 15-minute frequency.
        """
        if self.freq is None:
            raise ValueError("Frequency of the time series is not set.")
        return self.resample_to("15min")

    def resample_to(self, new_freq, method=None, agg="mean") -> "TimeseriesObject":
        """
        Resample the stored time series to a new frequency.

        :param new_freq: str
            The new frequency to resample to (e.g., '15min', 'H').
        :param method: str, optional
            The resampling method to use ('interpolate', 'ffill', 'bfill', or 'agg'). Defaults to None.
        :param agg: str, optional
            The aggregation function to use if method='agg' (e.g., 'mean', 'sum', 'last'). Defaults to 'mean'.
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
                resampled = self.data.resample(new_freq).interpolate("linear")
            elif method in ("ffill", "bfill"):
                resampled = getattr(self.data.resample(new_freq), method)()
            elif method == "agg":
                resampled = self.data.resample(new_freq).agg(agg)
            else:
                raise ValueError(
                    "Unsupported method. Use 'interpolate', 'ffill', 'bfill', or 'agg'."
                )
        except Exception as e:
            raise ValueError(f"Error during resampling: {e}")

        self.data = resampled
        return TimeseriesObject(data=resampled)

    def to_df(self) -> pd.DataFrame:
        """
        return: pd.DataFrame
            The original time series data.
        """
        return self.data

    def to_pulp(self, name: str, freq: str, time_set: int):
        """
        Convert the time series data to a format suitable for pulp optimization.

        If the data is empty, returns empty pulp variable with given name.
        If the data length does not match `time_set`, the data is resampled to the given frequency.
        Otherwise, returns the time series data as stored.

        :param name: int: The base name for pulp variables if data is empty.
        :param freq: int: The frequency to resample to if needed (e.g., '15min', '1H').
        :param time_set: int: The number of time steps in the time series.
        :return: Either a list of empty pulp variables or the DataFrame.
        """
        if self.data.empty:
            return [
                pulp.LpVariable(f"P_{name}_{t}", lowBound=0) for t in range(time_set)
            ]
        if time_set != len(self.data):
            return self.resample_to(freq).to_df()
        return self.to_df()
