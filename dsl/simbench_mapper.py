"""
tools/simbench_mapper.py

Helpers to map SimBench timeseries (CSV) into Omnes internal per-device CSV files.

Usage:
    - Provide simbench_ts_folder: folder containing SimBench CSV time series (one CSV per profile).
    - Provide a mapping dict mapping Omnes entity IDs -> simbench filename (or pandas.Series).
    - The function writes the Omnes-compatible CSV files to the provided out_folder
      and returns a dict entity_id -> pandas.Series for immediate use.

Notes:
    - This is intentionally simple and robust: it doesn't rely on any specific simbench
      Python API. If you prefer to use the simbench package directly, you can replace
      the file-reading with simbench API calls.
"""

import os
import pandas as pd
from typing import Dict, Union

def map_simbench_profiles(
    mapping: Dict[str, Union[str, pd.Series]],
    simbench_ts_folder: str,
    out_folder: str,
    time_col: str = None,
    time_format: str = None,
) -> Dict[str, pd.Series]:
    """
    mapping: dict where keys are Omnes entity ids (e.g., "pv_1", "load_2")
             and values are either:
               - the filename relative to simbench_ts_folder (e.g. "rural1_pv_01.csv")
               - a pandas.Series (already loaded)
    simbench_ts_folder: directory containing simbench CSV files (if mapping values are filenames)
    out_folder: directory where Omnes-compatible CSVs will be written (paths to be used in entity.input)
    time_col: optional - name of the column in simbench CSV that contains timestamps (if present)
    time_format: optional - strftime format for parsing time_col if needed
    Returns:
        dict entity_id -> pandas.Series indexed by pandas.DatetimeIndex (values in kW)
    """
    os.makedirs(out_folder, exist_ok=True)
    result = {}

    for entity_id, source in mapping.items():
        if isinstance(source, pd.Series):
            series = source.copy()
        elif isinstance(source, str):
            fname = os.path.join(simbench_ts_folder, source)
            if not os.path.exists(fname):
                raise FileNotFoundError(f"SimBench timeseries file not found: {fname}")
            df = pd.read_csv(fname)
            # try to locate numeric column automatically (simple heuristic)
            # if only two columns, assume second column is the value
            if time_col and time_col in df.columns:
                times = pd.to_datetime(df[time_col], format=time_format)
                vals = df.drop(columns=[time_col]).iloc[:, 0]
            elif df.shape[1] == 1:
                vals = df.iloc[:, 0]
                times = None
            else:
                # If first column is timestamp-like, detect it
                first = df.iloc[:, 0]
                try:
                    times = pd.to_datetime(first, infer_datetime_format=True)
                    vals = df.iloc[:, 1]
                except Exception:
                    # fallback: take the last column as values
                    times = None
                    vals = df.iloc[:, -1]

            series = pd.Series(vals.values, index=times)
        else:
            raise ValueError("mapping values must be filename or pandas.Series")

        # if index not datetime, keep as RangeIndex; but for Omnes we write a csv of values
        # normalize units: assume SimBench values are in kW (this is typical) - if not, user must scale
        # write to out_folder as CSV with a single column named 'value'
        out_path = os.path.join(out_folder, f"{entity_id}.csv")
        if series.index.is_all_dates:
            out_df = pd.DataFrame({"timestamp": series.index, "value": series.values})
            out_df.to_csv(out_path, index=False)
        else:
            out_df = pd.DataFrame({"value": series.values})
            out_df.to_csv(out_path, index=False)

        result[entity_id] = series

    return result


# Example usage:
if __name__ == "__main__":
    # sample mapping: user should adapt file names to actual SimBench csvs
    mapping = {
        "pv_1": "rural1_pv_01.csv",
        "load_1": "rural1_load_01.csv",
        "load_2": "rural1_load_02.csv",
    }
    simbench_folder = "/path/to/simbench/timeseries"
    out = "data/simbench_mapped"
    series_map = map_simbench_profiles(mapping, simbench_folder, out)
    print("Wrote files and loaded series for:", list(series_map.keys()))
