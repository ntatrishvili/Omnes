import datetime
import pandas as pd

class TimeseriesObject():
    
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize the timeseries object.
        """
        if data is None:
            self.data = pd.Series()
            self.freq: datetime.timedelta = None
            return
        self.data = pd.Series(data)
        self.data.index = pd.to_datetime(self.data.index)
        inferred = pd.infer_freq(self.data.index)
        self.freq: datetime.timedelta = self.normalize_freq(inferred)

    @classmethod
    def normalize_freq(cls, freq: str):
        """
        Normalize the frequency string to a standard format.
        - 'H' -> '1H'
        """
        if freq is not None and freq.isalpha():
            return '1' + freq
        return freq

    def to_1h(self):
        """
        Convert the time series to 1-hour frequency.

        Returns:
        - pd.Series with 1-hour frequency
        Raises ValueError if the frequency is not set.
        """
        if self.freq is None:
            raise ValueError("Frequency of the time series is not set.")
        return self.resample_to('1H')

    def to_15m(self):
        """
        Convert the time series to 15-minute frequency.

        Returns:
        - pd.Series with 15-minute frequency
        Raises ValueError if the frequency is not set.
        """
        if self.freq is None:
            raise ValueError("Frequency of the time series is not set.")
        return self.resample_to('15min')

    def resample_to(self, new_freq, method=None, agg='mean'):
        """
        Resample the stored time series to a new frequency.

        Parameters:
        - new_freq: str, like '15min', 'H', etc.
        - method: 'interpolate', 'ffill', 'bfill', or 'agg'
        - agg: aggregation function if method='agg' (e.g., 'mean', 'sum', 'last')

        Returns:
        - TimeseriesObject resampled to new_freq
        Raises ValueError if the frequency cannot be inferred or if an unsupported method is used.
        """
        current_freq = TimeseriesObject.normalize_freq(pd.infer_freq(self.data.index))
        if current_freq is None:
            raise ValueError("Cannot infer current frequency. Please specify method manually.")

        if method is None:
            if pd.Timedelta(new_freq) < pd.Timedelta(current_freq):
                method = 'interpolate'  # Upsampling
            else:
                method = 'agg'  # Downsampling

        if method == 'interpolate':
            self.data = self.data.resample(new_freq).interpolate('linear')
        elif method in ('ffill', 'bfill'):
            self.data = getattr(self.data.resample(new_freq), method)()
        elif method == 'agg':
            self.data = self.data.resample(new_freq).agg(agg)
        else:
            raise ValueError("Unsupported method. Use 'interpolate', 'ffill', 'bfill', or 'agg'.")
        return TimeseriesObject(self.data)

    def get_data(self):
        """Return the original time series."""
        return self.data
    