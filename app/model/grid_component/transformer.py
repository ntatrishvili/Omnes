from typing import Optional

from app.infra.timeseries_object_factory import (
    DefaultTimeseriesFactory,
    TimeseriesFactory,
)
from app.model.grid_component.grid_component import GridComponent


class Transformer(GridComponent):
    """
    Simple representation of a transformer (HV/LV) read from SimBench-like CSV.

    Attributes:
        hv_bus: identifier/name of the high-voltage bus
        lv_bus: identifier/name of the low-voltage bus
        type: descriptive type string from CSV (often contains SN MVA and kV levels)
        tappos: tap position (int)
        auto_tap: whether automatic tap is enabled (bool)
        auto_tap_side: which side auto tap applies to ("hv" or "lv" or None)
        loading_max: optional maximum loading value (float)
        substation: optional substation id/name
        subnet: optional subnet id/name
        volt_lvl: optional voltage level indicator from CSV
    """

    def __init__(
        self,
        id: Optional[str] = None,
        hv_bus: Optional[str] = None,
        lv_bus: Optional[str] = None,
        type: Optional[str] = None,
        tappos: Optional[int] = 0,
        auto_tap: Optional[bool] = False,
        auto_tap_side: Optional[str] = None,
        loading_max: Optional[float] = None,
        substation: Optional[str] = None,
        subnet: Optional[str] = None,
        volt_lvl: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs,
    ):
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        # Core mapping fields
        self.hv_bus = hv_bus or kwargs.pop("nodeHV", None)
        self.lv_bus = lv_bus or kwargs.pop("nodeLV", None)
        self.type = type or kwargs.pop("type", None)
        # tap & control
        self.tappos = int(tappos) if tappos is not None else 0
        self.auto_tap = bool(auto_tap)
        self.auto_tap_side = auto_tap_side or kwargs.pop("autoTapSide", None)
        # operational/meta
        self.loading_max = (
            float(loading_max)
            if loading_max is not None
            else kwargs.pop("loadingMax", None)
        )
        self.substation = substation or kwargs.pop("substation", None)
        self.subnet = subnet or kwargs.pop("subnet", None)
        self.volt_lvl = volt_lvl or kwargs.pop("voltLvl", None)

        # keep CSV/raw values in tags for debugging if present
        self.tags.update(
            {
                "type": self.type,
                "tappos": self.tappos,
                "auto_tap": self.auto_tap,
                "auto_tap_side": self.auto_tap_side,
                "loading_max": self.loading_max,
                "substation": self.substation,
                "subnet": self.subnet,
                "volt_lvl": self.volt_lvl,
            }
        )
