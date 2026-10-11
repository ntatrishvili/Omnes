"""Unmetered consumption = E.ON campus total - metered buildings - network losses."""

import pandas as pd

NOLOAD_LOSS_KW = 7.5  # 5 transformers x ~1.5 kW iron losses
LOSS_FRACTION = 0.01  # copper + cable losses as a share of the campus power


def residual_kw(
    campus_kw: pd.Series,
    metered_kw: pd.Series,
    noload_kw: float = NOLOAD_LOSS_KW,
    loss_fraction: float = LOSS_FRACTION,
) -> tuple[pd.Series, pd.Series]:
    """Unmetered load power (>= 0) and a flag for hours where it had to be clipped.

    Network losses are removed because the Pandapower simulation computes them itself.
    """
    losses = noload_kw + loss_fraction * campus_kw
    r = campus_kw - metered_kw - losses
    return r.clip(lower=0), r < 0


def allocate(residual: pd.Series, loads: pd.DataFrame) -> pd.DataFrame:
    """Split the residual over unmetered loads proportionally to their rated kVA.

    ``loads`` needs columns ``id`` and ``rated_kVA`` (Fig. 10 of the report). Returns
    one kW column per load id.
    """
    w = loads.set_index("id")["rated_kVA"]
    w = w / w.sum()
    return pd.DataFrame({i: residual * w[i] for i in w.index})
