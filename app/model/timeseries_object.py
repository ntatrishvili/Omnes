import datetime
import pandas as pd

from app.infra.util import get_input_path


class TimeseriesObject:
    """
    A class representing a time series object.
    This class is used to handle time series data, including reading from CSV files,
    resampling, and normalizing frequencies.
    """

    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize the timeseries object.
        :param data: pd.DataFrame, optional
            The data to initialize the TimeseriesObject. Defaults to None.
        """
        if data is None:
            self.data = pd.DataFrame()
            self.freq: datetime.timedelta = None
            return
        self.data = pd.DataFrame(data)
        self.data.index = pd.to_datetime(self.data.index)
        inferred = pd.infer_freq(self.data.index)
        self.freq: datetime.timedelta = self.normalize_freq(inferred)

    @staticmethod
    def read(filename: str, col: str) -> "TimeseriesObject":
        """
        Read a CSV file and return a TimeseriesObject with
        the specified column and timestamp parameters.

        :param filename: str Name of the input CSV file from data folder.
        :param col: str Name of the column to extract from the CSV file.
        :raises FileNotFoundError: If the specified file does not exist.
        :raises ValueError: If the file is empty or invalid.
        :raises KeyError: If the specified column is not found in the file.
        :return: TimeseriesObject A TimeseriesObject containing the specified column and timestamp as the index.
        """

        input_path = get_input_path(filename)
        try:
            input_df = pd.read_csv(
                input_path,
                sep=";",
                header=0,
                index_col="timestamp",
                parse_dates=["timestamp"],
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

        return TimeseriesObject(result)

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
        current_freq = TimeseriesObject.normalize_freq(pd.infer_freq(self.data.index))
        if current_freq is None:
            raise ValueError(
                "Cannot infer current frequency. Please specify method manually."
            )

        if method is None:
            if pd.Timedelta(new_freq) < pd.Timedelta(current_freq):
                method = "interpolate"  # Upsampling
            else:
                method = "agg"  # Downsampling

        if method == "interpolate":
            self.data = self.data.resample(new_freq).interpolate("linear")
        elif method in ("ffill", "bfill"):
            self.data = getattr(self.data.resample(new_freq), method)()
        elif method == "agg":
            self.data = self.data.resample(new_freq).agg(agg)
        else:
            raise ValueError(
                "Unsupported method. Use 'interpolate', 'ffill', 'bfill', or 'agg'."
            )
        return TimeseriesObject(self.data)

    def get_data(self) -> pd.DataFrame:
        """
        return: pd.DataFrame
            The original time series data.
        """
        return self.data
