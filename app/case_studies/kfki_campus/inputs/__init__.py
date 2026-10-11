"""Cleaning and assembly of the real KFKI input time series.

Raw files in ``data/input`` (LiveView daily meters, E.ON 15-min campus total,
weather logger, meter register) -> one hourly CSV in ``data/processed`` that
follows the README contract of the case study.

All outputs use one clock: fixed UTC+1 (CET all year, no daylight saving).
"""
