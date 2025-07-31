import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, List, Dict, Any

from app.infra.quantity import Quantity


def infer_freq_from_two_dates(data):
    """Helper function to infer frequency from two dates, supports both pandas and xarray."""
    if isinstance(data, xr.DataArray):
        time_coord = data.coords['time']
        delta = time_coord[1] - time_coord[0]
        # Convert xarray timedelta to pandas timedelta for total_seconds()
        seconds = pd.Timedelta(delta.values).total_seconds()
    else:
        delta = data.index[1] - data.index[0]
        seconds = delta.total_seconds()

    if seconds % 3600 == 0:
        freq_str = f"{int(seconds // 3600)}h"
    elif seconds % 60 == 0:
        freq_str = f"{int(seconds // 60)}min"
    else:
        freq_str = f"{int(seconds)}s"

    return freq_str


class TimeseriesObject(Quantity):
    """
    A class representing a time series object using xarray.DataArray as the backend.
    This class handles multi-dimensional time series data, including reading from CSV files,
    resampling, and normalizing frequencies. Maintains full backward compatibility with 
    pandas DataFrame-based code while enabling multi-dimensional energy system modeling.
    """

    def __init__(self, **kwargs):
        """
        Initialize the TimeseriesObject with either:
        - an xarray DataArray (using the ``data`` keyword)
        - a pandas DataFrame (using the ``data`` keyword) - auto-converted to xarray
        - a CSV file and column name (using the ``input_path`` and ``col`` keywords).

        If neither is provided, an empty DataArray is used.

        :keyword data: The xarray DataArray or pandas DataFrame to initialize with.
        :type data: xr.DataArray or pd.DataFrame, optional
        :keyword input_path: The path to the CSV file to read data from.
        :type input_path: str, optional
        :keyword col: The column name to extract from the CSV file.
        :type col: str, optional
        :keyword freq: str, optional
        :keyword dims: Dimension names for multi-dimensional data.
        :type dims: list, optional
        :keyword coords: Coordinate dictionaries for xarray.
        :type coords: dict, optional
        :keyword attrs: Metadata attributes for xarray.
        :type attrs: dict, optional
        """
        super().__init__(**kwargs)
        
        # Extract and prepare parameters
        params = self._extract_init_parameters(kwargs)
        
        # Initialize data
        self.data = self._initialize_data_array(params)
        
        # Set frequency
        self.freq = self._initialize_frequency(params['freq'])

    def _extract_init_parameters(self, kwargs):
        """Extract and prepare initialization parameters."""
        attrs = kwargs.pop("attrs", {})
        
        # Add remaining kwargs as metadata attributes
        for key, value in kwargs.items():
            if not key.startswith('_'):
                attrs[key] = value
        
        return {
            'data': kwargs.pop("data", None),
            'input_path': kwargs.pop("input_path", None),
            'col': kwargs.pop("col", None),
            'freq': kwargs.pop("freq", None),
            'dims': kwargs.pop("dims", None),
            'coords': kwargs.pop("coords", None),
            'attrs': attrs
        }

    def _initialize_data_array(self, params):
        """Initialize the xarray DataArray based on input parameters."""
        if isinstance(params['data'], xr.DataArray):
            data = params['data'].copy()
            data.attrs.update(params['attrs'])
            return data
        elif isinstance(params['data'], pd.DataFrame):
            return self._dataframe_to_xarray(params['data'], params['dims'], params['coords'], params['attrs'])
        elif params['input_path'] is not None and params['col'] is not None:
            df_data = self._read_csv_to_dataframe(params['input_path'], params['col'])
            return self._dataframe_to_xarray(df_data, params['dims'], params['coords'], params['attrs'])
        else:
            return xr.DataArray(
                data=[],
                dims=['time'],
                coords={'time': []},
                attrs=params['attrs']
            )

    def _initialize_frequency(self, freq_param):
        """Initialize the frequency attribute."""
        if self.empty():
            return None
        
        if freq_param is not None:
            self.resample_to(freq_param)
            return freq_param
        
        return self._infer_frequency_from_data()

    def _infer_frequency_from_data(self):
        """Infer frequency from the data."""
        if self.data.sizes['time'] < 3:
            return infer_freq_from_two_dates(self.data)
        else:
            return self.normalize_freq(
                pd.infer_freq(pd.DatetimeIndex(self.data.coords['time'].values))
            )

    def _dataframe_to_xarray(self, df: pd.DataFrame, dims: Optional[List[str]] = None, 
                            coords: Optional[Dict[str, Any]] = None, 
                            attrs: Optional[Dict[str, Any]] = None) -> xr.DataArray:
        """Convert pandas DataFrame to xarray DataArray."""
        if attrs is None:
            attrs = {}
        
        # For single-column DataFrames (most common case)
        if len(df.columns) == 1:
            col_name = df.columns[0]
            return xr.DataArray(
                data=df[col_name].values,
                dims=['time'],
                coords={'time': df.index},
                attrs=attrs,
                name=col_name
            )
        else:
            # Multi-column DataFrame - create variable dimension
            return xr.DataArray(
                data=df.values,
                dims=['time', 'variable'],
                coords={'time': df.index, 'variable': df.columns},
                attrs=attrs
            )

    def _read_csv_to_dataframe(self, input_path: str, col: str) -> pd.DataFrame:
        """Helper method to read CSV file and return DataFrame."""
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

        return input_df[[col]]

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
        ts = TimeseriesObject()
        df_data = ts._read_csv_to_dataframe(input_path, col)
        return TimeseriesObject(data=df_data)

    @classmethod
    def normalize_freq(cls, freq: str) -> str:
        """
        Normalize the frequency string to a standard format.
        :param freq: str
            The frequency string to normalize (e.g., 'h').
        :return: str
            The normalized frequency string (e.g., '1h').
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
        return self.resample_to("1h", closed=closed)

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
        self,
        new_freq,
        method=None,
        agg="mean",
        closed="right",
        in_place=False,
        keep_original_dtypes=False,
    ) -> "TimeseriesObject":
        """
        Resample the stored time series to a new frequency.

        :param new_freq: str
            The new frequency to resample to (e.g., '15min', 'h').
        :param method: str, optional
            The resampling method to use ('interpolate', 'ffill', 'bfill', or 'agg'). Defaults to None.
        :param agg: str, optional
            The aggregation function to use if method='agg' (e.g., 'mean', 'sum', 'last'). Defaults to 'mean'.
        :param closed: str, optional
            Which side of bin interval is closed. ('right', 'left'). Defaults to 'right'.
        :param in_place: bool, optional
            Signals whether the object itself is modified at the end of the operation
        :param keep_original_dtypes: bool, optional
            Signals whether the returned object's stored data types correspond to the datatypes of the original obhect.
        :raises ValueError:
            If the frequency cannot be inferred or if an unsupported method is used.
        :return: TimeseriesObject
            A TimeseriesObject resampled to the specified frequency.
        """
        if self.empty() or self.freq == new_freq:
            return self

        print(f"Resampling from {self.freq} to {new_freq}")
        
        # Get resampled DataFrame
        resampled_df = self._perform_resampling(new_freq, method, agg, closed, keep_original_dtypes)
        
        # Convert back to xarray and create result
        return self._create_resampled_result(resampled_df, new_freq, in_place)

    def _perform_resampling(self, new_freq, method, agg, closed, keep_original_dtypes):
        """Perform the actual resampling operation."""
        try:
            # Convert to pandas for resampling (xarray resampling is more limited)
            df = self.to_df()
            current_freq = self._validate_and_get_current_freq(df)
            
            # Determine resampling method
            method = self._determine_resampling_method(method, new_freq, current_freq)
            
            # Perform resampling
            resampled = self._resample_dataframe(df, new_freq, method, agg, closed)
            
            # Apply dtype preservation if requested
            if keep_original_dtypes:
                resampled = resampled.astype(df.dtypes)
                
            return resampled
            
        except Exception as e:
            raise ValueError(f"Error during resampling: {e}")

    def _validate_and_get_current_freq(self, df):
        """Validate and get the current frequency from DataFrame."""
        try:
            current_freq = TimeseriesObject.normalize_freq(pd.infer_freq(df.index))
        except Exception as e:
            raise ValueError(f"Error inferring frequency: {e}")
        
        if current_freq is None:
            raise ValueError("Cannot infer current frequency. Please specify method manually.")
        
        return current_freq

    def _determine_resampling_method(self, method, new_freq, current_freq):
        """Determine the appropriate resampling method."""
        if method is None:
            if pd.Timedelta(new_freq) < pd.Timedelta(current_freq):
                return "interpolate"  # Upsampling
            else:
                return "agg"  # Downsampling
        return method

    def _resample_dataframe(self, df, new_freq, method, agg, closed):
        """Resample the DataFrame using the specified method."""
        resampler = df.resample(new_freq, closed=closed)
        
        if method == "interpolate":
            return resampler.interpolate("linear")
        elif method in ("ffill", "bfill"):
            return getattr(resampler, method)()
        elif method == "agg":
            return resampler.agg(agg)
        else:
            raise ValueError("Unsupported method. Use 'interpolate', 'ffill', 'bfill', or 'agg'.")

    def _create_resampled_result(self, resampled_df, new_freq, in_place):
        """Create the final resampled TimeseriesObject result."""
        # Convert back to xarray DataArray
        resampled_data = self._dataframe_to_xarray(
            resampled_df, 
            list(self.data.dims), 
            None, 
            self.data.attrs
        )
        
        if in_place:
            self.data = resampled_data
            self.freq = new_freq
            return self
        else:
            result = TimeseriesObject(data=resampled_data)
            result.freq = new_freq
            return result

    def to_df(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for backward compatibility."""
        if 'variable' in self.data.dims:
            # Multi-variable case created from DataFrame
            return self.data.to_pandas()
        elif len(self.data.dims) == 1:
            # Single dimension case (time only)
            return pd.DataFrame(
                {self.data.name or 'value': self.data.values},
                index=self.data.coords['time'].values
            )
        else:
            # Multi-dimensional case - flatten non-time dimensions and create columns
            # This provides a basic backward compatibility for multi-dimensional data
            time_coord = self.data.coords['time']
            
            # Reshape the data: keep time as first dimension, flatten others
            time_size = self.data.sizes['time']
            reshaped_data = self.data.values.reshape(time_size, -1)
            
            # Create column names based on coordinate combinations
            non_time_dims = [dim for dim in self.data.dims if dim != 'time']
            if len(non_time_dims) == 1:
                # Single non-time dimension
                dim_name = non_time_dims[0]
                coord_values = self.data.coords[dim_name].values
                columns = [f"{dim_name}_{val}" for val in coord_values]
            else:
                # Multiple non-time dimensions - enumerate columns
                columns = [f"col_{i}" for i in range(reshaped_data.shape[1])]
            
            return pd.DataFrame(
                reshaped_data,
                index=time_coord.values,
                columns=columns
            )

    def to_nd(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data.values.flatten()

    def get_values(self, **kwargs):
        """Get values with optional resampling and slicing."""
        freq = kwargs.get("freq", self.freq)
        time_set = kwargs.get("time_set", self.data.sizes.get('time', 0))
        
        if freq != self.freq:
            resampled = self.resample_to(freq)
            if time_set != resampled.data.sizes.get('time', 0):
                return resampled.data.values.flatten()[:time_set]
            return resampled.to_nd()
        
        if time_set != self.data.sizes.get('time', 0):
            return self.data.values.flatten()[:time_set]
        return self.to_nd()

    # New xarray-specific methods
    def sel(self, **kwargs) -> "TimeseriesObject":
        """Select data along dimensions (xarray-style selection)."""
        selected_data = self.data.sel(**kwargs)
        result = TimeseriesObject(data=selected_data)
        result.freq = self.freq
        return result

    def isel(self, **kwargs) -> "TimeseriesObject":
        """Select data by integer indices."""
        selected_data = self.data.isel(**kwargs)
        result = TimeseriesObject(data=selected_data)
        result.freq = self.freq
        return result

    def add_dimension(self, dim_name: str, coord_values: List[Any], 
                     axis: Optional[int] = None) -> "TimeseriesObject":
        """Add a new dimension to the data."""
        expanded_data = self.data.expand_dims({dim_name: coord_values}, axis=axis)
        result = TimeseriesObject(data=expanded_data)
        result.freq = self.freq
        return result

    def set_metadata(self, **attrs) -> "TimeseriesObject":
        """Set metadata attributes."""
        new_data = self.data.copy()
        new_data.attrs.update(attrs)
        result = TimeseriesObject(data=new_data)
        result.freq = self.freq
        return result

    def get_metadata(self, key: str = None):
        """Get metadata attributes."""
        if key is None:
            return dict(self.data.attrs)
        return self.data.attrs.get(key)

    def __getattr__(self, name):
        """Delegate attribute access to xarray DataArray."""
        data_attr = getattr(self.data, name, None)
        if data_attr is not None:
            return data_attr
        raise AttributeError(f"'TimeseriesObject' object has no attribute '{name}'")

    def empty(self) -> bool:
        """Check if the data is empty."""
        return self.data.sizes.get('time', 0) == 0

    def __eq__(self, other):
        """Check equality with another TimeseriesObject."""
        if not isinstance(other, TimeseriesObject):
            return False
        return self.data.equals(other.data) and self.freq == other.freq

    def __repr__(self):
        """String representation."""
        return f"TimeseriesObject(dims={self.data.dims}, shape={self.data.shape}, freq='{self.freq}')"
