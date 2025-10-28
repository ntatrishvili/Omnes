from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from app.infra.quantity import Quantity


def infer_freq_from_two_dates(data: xr.DataArray) -> str:
    """Infer timeseries frequency from first two timestamps.

    :param xr.DataArray data: DataArray with 'timestamp' coordinate
    :returns str: Frequency string (e.g. '1h', '15min')
    :raises ValueError: If 'timestamp' coordinate is missing
    """
    if "timestamp" not in data.coords:
        raise ValueError(
            "The provided DataArray does not contain a 'timestamp' coordinate."
        )
    time_coord = data.coords["timestamp"]
    delta = time_coord[1] - time_coord[0]
    # Convert xarray timedelta to pandas timedelta for total_seconds()
    seconds = pd.Timedelta(delta.values).total_seconds()

    if seconds % 3600 == 0:
        freq_str = f"{int(seconds // 3600)}h"
    elif seconds % 60 == 0:
        freq_str = f"{int(seconds // 60)}min"
    else:
        freq_str = f"{int(seconds)}s"

    return freq_str


class TimeseriesObject(Quantity):
    """A class representing a time series object using xarray.DataArray as the backend.
    This class handles multi-dimensional time series data, including reading from CSV files,
    resampling, and normalizing frequencies. Maintains full backward compatibility with
    pandas DataFrame-based code while enabling multi-dimensional energy system modeling.

    :ivar xr.DataArray data: Underlying time series data
    :ivar str freq: Frequency string (e.g. '1h')
    """

    def __init__(self, **kwargs):
        """
        Initialize from xarray, pandas, or CSV file.

        :param xr.DataArray|pd.DataFrame data: DataArray or DataFrame (optional)
        :param str input_path: Path to CSV (optional)
        :param str col: Column name (optional)
        :param str freq: Frequency string (optional)
        """
        super().__init__(**kwargs)

        self.freq = None

        params = self._extract_init_parameters(kwargs)

        self.data = self._initialize_data_array(params)

        self.freq = self._initialize_frequency(params["freq"])

    def _extract_init_parameters(self, kwargs):
        """Extract and prepare initialization parameters.

        :param dict kwargs: Keyword arguments
        :returns dict: Processed parameters
        """
        attrs = kwargs.pop("attrs", {})

        # Add remaining kwargs as metadata attributes
        for key, value in kwargs.items():
            if not key.startswith("_"):
                attrs[key] = value

        return {
            "data": kwargs.pop("data", None),
            "input_path": kwargs.pop("input_path", None),
            "col": kwargs.pop("col", None),
            "datetime_column": kwargs.pop("datetime_column", None),
            "datetime_format": kwargs.pop("datetime_format", None),
            "tz": kwargs.pop("tz", None),
            "freq": kwargs.pop("freq", None),
            "dims": kwargs.pop("dims", None),
            "coords": kwargs.pop("coords", None),
            "attrs": attrs,
        }

    def _initialize_data_array(self, params):
        """Initialize DataArray from parameters.

        :param dict params: Initialization parameters
        :returns xr.DataArray: DataArray
        """
        if isinstance(params["data"], xr.DataArray):
            data = params["data"].copy()
            data.attrs.update(params["attrs"])
            return data
        elif isinstance(params["data"], pd.DataFrame):
            return self._dataframe_to_xarray(params["data"], params["attrs"])
        elif params["input_path"] is not None and params["col"] is not None:
            df_data = self._read_csv_to_dataframe(
                params["input_path"],
                params["col"],
                datatime_column=params.get("datetime_column", None),
                datetime_format=params.get("datetime_format", None),
                tz=params.get("tz", None),
            )
            return self._dataframe_to_xarray(df_data, params["attrs"])
        else:
            return xr.DataArray(data=[], dims=["timestamp"], attrs=params["attrs"])

    def _initialize_frequency(self, freq_param):
        """Initialize frequency attribute.

        :param str|None freq_param: Frequency string or None
        :returns str|None: Frequency string or None
        """
        if self.empty():
            return None

        if freq_param is not None:
            self.resample_to(freq_param, in_place=True)
            return freq_param

        return self._infer_frequency_from_data()

    def _infer_frequency_from_data(self):
        """Infer frequency from data.

        :returns str: Inferred frequency string
        """
        if self.data.sizes.get("timestamp", 0) < 3:
            return infer_freq_from_two_dates(self.data)
        # Use xarray's infer_freq directly on the timestamp coordinate
        freq = xr.infer_freq(self.data.coords["timestamp"].values)
        return self.normalize_freq(freq)

    def _dataframe_to_xarray(
        self,
        df: pd.DataFrame,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> xr.DataArray:
        """Convert DataFrame to DataArray.

        :param pd.DataFrame df: DataFrame
        :param dict|None attrs: Metadata attributes (optional)
        :returns xr.DataArray: DataArray
        """
        if attrs is None:
            attrs = {}

        # For single-column DataFrames
        if len(df.columns) == 1:
            col_name = df.columns[0]
            return xr.DataArray(
                data=df[col_name].values,
                dims=["timestamp"],
                coords={"timestamp": df.index},
                attrs=attrs,
                name=col_name,
            )
        else:
            # Multi-column DataFrame - create variable dimension
            return xr.DataArray(
                data=df.values,
                dims=["timestamp", "variable"],
                coords={"timestamp": df.index, "variable": df.columns},
                attrs=attrs,
            )

    @staticmethod
    def _read_csv_to_dataframe(
        input_path: str,
        col: str,
        datatime_column: Optional[str] = None,
        datetime_format: Optional[str] = None,
        tz: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read CSV and return DataFrame with specified column and parsed timestamp index.

        :param str input_path: Path to CSV
        :param str col: Column name to return
        :param str|None datatime_column: Name of time column to parse (default 'timestamp').
                                  If None, the function will try to auto-detect a datetime-like column.
        :param str|None datetime_format: Optional datetime format string to use for parsing.
        :returns pd.DataFrame: DataFrame with column and timestamp index
        :raises FileNotFoundError: If file does not exist
        :raises ValueError: If file is empty or invalid or no datetime column can be parsed
        :raises KeyError: If requested column not found
        """
        try:
            input_df = pd.read_csv(input_path, sep=";", header=0)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{input_path}' does not exist.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file '{input_path}' is empty or invalid.")

        if col not in input_df.columns:
            raise KeyError(
                f"The column '{col}' is not found in the file {input_path}'. {input_df}"
            )

        # If a specific time column was provided
        if datatime_column is not None:
            if datatime_column not in input_df.columns:
                raise KeyError(
                    f"The time column '{datatime_column}' is not found in the file {input_path}. Columns: {list(input_df.columns)}"
                )
            # Try parsing using provided format first (if any), then fallback to automatic parsing
            input_df = TimeseriesObject.__parse_time_col(
                input_df, datatime_column, datetime_format=datetime_format
            )
        else:
            found_col = False
            # Auto-detect a datetime-like column (choose the first column where parsing yields many non-nulls)
            for c in input_df.columns:
                if "time" not in c.lower() and "date" not in c.lower():
                    continue
                input_df = TimeseriesObject.__parse_time_col(
                    input_df, c, datetime_format=datetime_format
                )
                found_col = True
                break
            if not found_col:
                raise ValueError(
                    f"No datetime-like column could be auto-detected in the file '{input_path}'."
                )
        # Localize timezone if specified
        if tz is not None:
            input_df.index = input_df.index.tz_localize(tz, ambiguous="infer")
        return input_df[[col]]

    @staticmethod
    def __parse_time_col(input_df, time_col, datetime_format=None):
        input_df[time_col] = pd.to_datetime(
            input_df[time_col], format=datetime_format, errors="coerce"
        )
        if input_df[time_col].isna().all():
            raise ValueError(
                f"Could not parse datetime values from column '{time_col}'."
            )
        input_df = input_df.rename(columns={time_col: "timestamp"})
        input_df.set_index("timestamp", inplace=True, drop=True)
        input_df.index = pd.to_datetime(input_df.index)
        return input_df

    @staticmethod
    def read(
        input_path: str,
        col: str,
        time_col: Optional[str] = "timestamp",
        datetime_format: Optional[str] = None,
    ) -> "TimeseriesObject":
        """Read CSV and return TimeseriesObject.

        :param str input_path: Path to CSV
        :param str col: Name of the column to read
        :param str|None time_col: Name of the time column in the CSV (default 'timestamp'). If None, auto-detect.
        :param str|None datetime_format: Optional datetime format to use for parsing
        :returns TimeseriesObject: TimeseriesObject with column and timestamp index
        """
        return TimeseriesObject(
            data=TimeseriesObject._read_csv_to_dataframe(
                input_path,
                col,
                datatime_column=time_col,
                datetime_format=datetime_format,
            )
        )

    @classmethod
    def normalize_freq(cls, freq: str) -> Optional[str]:
        """Normalize frequency string.

        :param str freq: Frequency string
        :returns str|None: Normalized frequency string (e.g. '1h') or None
        """
        if freq is None:
            return None
        if freq.isalpha():
            return f"1{freq}"
        return freq

    def to_1h(self, closed="left") -> "TimeseriesObject":
        """Convert to 1-hour frequency.

        :param str closed: Which side of bin interval is closed (default 'left')
        :returns TimeseriesObject: Resampled TimeseriesObject
        :raises ValueError: If frequency is not set
        """
        if self.freq is None:
            raise ValueError("Frequency of the time series is not set.")
        return self.resample_to("1h")

    def to_15m(self, closed="left") -> "TimeseriesObject":
        """Convert to 15-minute frequency.

        :param str closed: Which side of bin interval is closed (default 'left')
        :returns TimeseriesObject: Resampled TimeseriesObject
        :raises ValueError: If frequency is not set
        """
        if self.freq is None:
            raise ValueError("Frequency of the time series is not set.")
        return self.resample_to("15min")

    def resample_to(
        self,
        new_freq,
        method=None,
        agg="sum",
        in_place=False,
    ) -> "TimeseriesObject":
        """Resample to new frequency.

        :param str new_freq: New frequency (e.g. '15min')
        :param str|None method: Resampling method (optional)
        :param str agg: Aggregation function (default 'sum')
        :param bool in_place: Modify in place (default False)
        :returns TimeseriesObject: Resampled TimeseriesObject
        :raises ValueError: If frequency cannot be inferred or method unsupported
        """
        if self.empty() or self.freq == new_freq:
            return self

        try:
            current_freq = self._infer_frequency_from_data()
        except Exception as e:
            raise ValueError(f"Error inferring current frequency: {e}")

        if method is None:
            if pd.Timedelta(new_freq) < pd.Timedelta(current_freq):
                method = "interpolate"  # Upsampling
            else:
                method = "agg"  # Downsampling

        resampler = self.data.resample(timestamp=new_freq)
        if method == "interpolate":
            resampled = resampler.interpolate("linear")
            ratio = self.data.sum(skipna=True) / resampled.sum(skipna=True)
            resampled *= ratio
        elif method == "agg":
            resampled = getattr(resampler, agg)()
        else:
            raise ValueError("Unsupported method. Use 'interpolate' or 'agg'.")

        if in_place:
            self.data = resampled
            self.freq = new_freq
            return self
        result = TimeseriesObject(data=resampled)
        result.freq = new_freq
        return result

    def to_df(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.

        :returns pd.DataFrame: DataFrame representation
        """
        if "variable" in self.data.dims:
            # Multi-variable case created from DataFrame
            return self.data.to_pandas()
        elif len(self.data.dims) == 1:
            # Single dimension case (timestamp only)
            return pd.DataFrame(
                {self.data.name or "value": self.data.values},
                index=pd.DatetimeIndex(self.data.coords["timestamp"].values),
            )
        else:
            # Multi-dimensional case - flatten non-timestamp dimensions and create columns
            # This provides a basic backward compatibility for multi-dimensional data
            timestamp_coord = self.data.coords["timestamp"]

            # Reshape the data: keep timestamp as first dimension, flatten others
            timestamp_size = self.data.sizes["timestamp"]
            reshaped_data = self.data.values.reshape(timestamp_size, -1)

            # Create column names based on coordinate combinations
            non_timestamp_dims = [dim for dim in self.data.dims if dim != "timestamp"]
            if len(non_timestamp_dims) == 1:
                # Single non-timestamp dimension
                dim_name = non_timestamp_dims[0]
                coord_values = self.data.coords[dim_name].values
                columns = [f"{dim_name}_{val}" for val in coord_values]
            else:
                # Multiple non-timestamp dimensions - enumerate columns
                columns = [f"col_{i}" for i in range(reshaped_data.shape[1])]

            return pd.DataFrame(
                reshaped_data,
                index=pd.DatetimeIndex(timestamp_coord.values),
                columns=columns,
            )

    def to_nd(self) -> np.ndarray:
        """Convert to numpy array.

        :returns np.ndarray: Flattened numpy array
        """
        return self.data.values.flatten()

    def value(self, **kwargs):
        """Get values with optional resampling and slicing.

        :param str freq: Frequency to resample to (optional)
        :param int time_set: Number of time steps (optional)
        :returns np.ndarray: Array of values
        """
        freq = kwargs.get("freq", self.freq)
        time_set = kwargs.get("time_set", self.data.sizes.get("timestamp", 0))

        if freq != self.freq:
            resampled = self.resample_to(freq)
            if time_set != resampled.data.sizes.get("timestamp", 0):
                return resampled.data.values.flatten()[:time_set]
            return resampled.to_nd()

        if time_set != self.data.sizes.get("timestamp", 0):
            return self.data.values.flatten()[:time_set]
        return self.to_nd()

    # New xarray-specific methods
    def sel(self, **kwargs) -> "TimeseriesObject":
        """Select data along dimensions.

        :param kwargs: Dimension names and values
        :returns TimeseriesObject: New TimeseriesObject with selected data
        """
        selected_data = self.data.sel(**kwargs)
        result = TimeseriesObject(data=selected_data)
        result.freq = self.freq
        return result

    def isel(self, **kwargs) -> "TimeseriesObject":
        """Select data by integer indices.

        :param kwargs: Dimension names and indices
        :returns TimeseriesObject: New TimeseriesObject with selected data
        """
        selected_data = self.data.isel(**kwargs)
        result = TimeseriesObject(data=selected_data)
        result.freq = self.freq
        return result

    def add_dimension(
        self, dim_name: str, coord_values: List[Any], axis: Optional[int] = None
    ) -> "TimeseriesObject":
        """Add a new dimension.

        :param str dim_name: Name of new dimension
        :param list coord_values: Coordinate values
        :param int|None axis: Axis to add (optional)
        :returns TimeseriesObject: New TimeseriesObject
        """
        expanded_data = self.data.expand_dims({dim_name: coord_values}, axis=axis)
        result = TimeseriesObject(data=expanded_data)
        result.freq = self.freq
        return result

    def set_metadata(self, **attrs) -> "TimeseriesObject":
        """Set metadata attributes.

        :param attrs: Metadata attributes
        :returns TimeseriesObject: New TimeseriesObject with updated metadata
        """
        new_data = self.data.copy()
        new_data.attrs.update(attrs)
        result = TimeseriesObject(data=new_data)
        result.freq = self.freq
        return result

    def get_metadata(self, key: str = None):
        """Get metadata attributes.

        :param str|None key: Metadata key (optional)
        :returns dict|Any: All attributes or value for key
        """
        if key is None:
            return dict(self.data.attrs)
        return self.data.attrs.get(key)

    def __getattr__(self, name):
        """Delegate attribute access to the underlying DataArray when safe.

        Avoid recursion when `self.data` is another TimeseriesObject or is None,
        and do not delegate for core attributes.
        """
        # prevent delegation for core attributes
        if name in ("data", "freq"):
            raise AttributeError(f"'TimeseriesObject' object has no attribute '{name}'")

        # get the underlying data without triggering __getattr__
        data = object.__getattribute__(self, "data")

        # if there's no underlying object or it's another TimeseriesObject, do not delegate
        if data is None or isinstance(data, TimeseriesObject):
            raise AttributeError(f"'TimeseriesObject' object has no attribute '{name}'")

        # delegate to the underlying object when safe
        data_attr = getattr(data, name, None)
        if data_attr is not None:
            return data_attr

        raise AttributeError(f"'TimeseriesObject' object has no attribute '{name}'")

    def empty(self) -> bool:
        """Check if data is empty.

        :returns bool: True if empty, else False
        """
        return self.data.sizes.get("timestamp", 0) == 0

    def __eq__(self, other):
        """Check equality with another TimeseriesObject.

        :param TimeseriesObject other: Object to compare
        :returns bool: True if equal, else False
        """
        if not isinstance(other, TimeseriesObject):
            return False
        return self.data.equals(other.data) and self.freq == other.freq

    def __repr__(self):
        """String representation.

        :returns str: String representation
        """
        return f"TimeseriesObject(dims={self.data.dims}, shape={self.data.shape}, freq='{self.freq}')"
