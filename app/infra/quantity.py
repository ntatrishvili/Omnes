from abc import ABC, abstractmethod


class Direction:
    IN = "in"
    OUT = "out"


class RunData:
    """Container for data specific to one optimization run."""
    def __init__(self):
        self.aligned = None      # Resampled data (TimeSet-aligned)
        self.result = None       # Optimization results (same shape as aligned)
        self._vars = None        # Solver variables (temporary, MUST be cleared)


class Quantity(ABC):
    """
    Abstract base class for representing any abstract quantity (e.g. power flow, financial amount) within an entity.

    Quantities may represent static scalar parameters (Parameter) or dynamic time series (TimeSeriesObject).
    This class defines the interface for how quantities are integrated into
    the optimization model.

    Attributes:
        - Implementations should store any metadata or values passed via **kwargs.
    """

    def __init__(self, **kwargs):
        self._runs: dict[str, RunData] = {}  # run_id → RunData
        self.direction = kwargs.pop("direction", None)

    def convert(self, converter, **kwargs):
        """Converts the quantity into a pulp-compatible format (e.g., a time series array or a value-variable)."""
        return converter.convert_quantity(self, **kwargs)

    def set(self, value, **kwargs):
        """Sets the value of the quantity, if applicable."""
        ...

    def set_value(self, value, **kwargs):
        return self.set(value, **kwargs)


    def run(self, run_id: str) -> RunData:
        """Get or create RunData for this run.
        
        :param str run_id: Unique identifier for this optimization run
        :returns RunData: Container for run-specific data
        """
        if run_id not in self._runs:
            self._runs[run_id] = RunData()
        return self._runs[run_id]

    def clear_run(self, run_id: str) -> None:
        """Clean up run data (use after optimization complete).
        
        :param str run_id: Unique identifier for the run to clear
        """
        if run_id in self._runs:
            del self._runs[run_id]

    def list_runs(self) -> list[str]:
        """For debugging: show which runs have data.
        
        :returns list[str]: List of run IDs with stored data
        """
        return list(self._runs.keys())

    @property
    def value(self, **kwargs):  # NOSONAR
        return None

    @abstractmethod
    def __eq__(self, other): ...

    @abstractmethod
    def empty(self) -> bool: ...
