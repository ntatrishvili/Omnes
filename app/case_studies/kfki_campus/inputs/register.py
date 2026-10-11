"""Meter register (``KFKI_merok_kapcsolatai_*.xlsx``, sheet ``Elektromos``).

The sheet is a hierarchy: transformer -> main meter -> sub-meter -> sub-sub-meter.
The main meters with 3670xxxx serials are the LiveView meters.
"""

import re
from pathlib import Path

import pandas as pd

# register transformer name -> id used in the model (diagram names)
TRAFO_NAME = {
    "6-os": "6T",
    "3-as": "3T",
    "4-es": "4T",
    "IKI": "T",
    "Reaktor": "10T",
}
# columns (by position) of [serial, building, place] per level
LEVELS = {"main1": 7, "main2": 10, "sub": 13, "subsub": 16}


def _trafo_id(name: str) -> str:
    for key, tid in TRAFO_NAME.items():
        if str(name).strip().startswith(key):
            return tid
    return ""


def read_register(path: Path) -> pd.DataFrame:
    """Tidy table: one row per (meter, level) with transformer, building and place."""
    raw = pd.read_excel(path, sheet_name="Elektromos", header=0, dtype=str)
    rows = []
    for _, r in raw.iterrows():
        v = r.tolist()
        tid = _trafo_id(v[6])
        for level, c in LEVELS.items():
            serial = str(v[c]).strip()
            if serial in ("", "nan"):
                continue
            rows.append(
                {
                    "meter": serial,
                    "level": level,
                    "building": str(v[c + 1]).strip(),
                    "place": str(v[c + 2]).strip(),
                    "trafo": tid,
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(["meter", "level", "building"])
    return df.reset_index(drop=True)


# model loads without a building number -> register building label (prefix, case-insensitive)
NAMED_LOADS = {
    "reaktor_1": "10",  # 10-es trafohaz (reactor transformer)
    "reaktor_2": "10",
    "sportpalya": "Sporttelep",
    "fahaz_1_3": "faház",
}


def matches_building(register_building: str, label: str) -> bool:
    """'22' matches 22, 22A, 22B, 22/1 (not 220); a named label matches by prefix."""
    rb, lb = register_building.strip().lower(), label.strip().lower()
    if not lb:
        return False
    if lb[0].isalpha():
        return rb.startswith(lb)
    return re.fullmatch(rf"{re.escape(lb)}([/a-z].*)?", rb) is not None


def building_of_load(load_id: str, building: object) -> str:
    """Register-style building label of a model load ('b23_5' -> '23/5')."""
    if load_id in NAMED_LOADS:
        return NAMED_LOADS[load_id]
    b = "" if pd.isna(building) else str(building).strip()
    if b:
        return b
    if load_id.startswith("b") and load_id[1:2].isdigit():
        return load_id[1:].replace("_", "/")
    return ""


def download_priority(
    loads: pd.DataFrame, register: pd.DataFrame, metered_loads: set[str]
) -> pd.DataFrame:
    """Unmetered model loads ranked by rated power, with candidate register meters.

    ``loads`` is the ``loads`` sheet of assets.xlsx. Use it to decide which LiveView
    meters to download next.
    """
    main = register[register["level"].isin(["main1", "main2"])]
    rows = []
    for r in loads.itertuples():
        if r.id in metered_loads:
            continue
        bld = building_of_load(r.id, r.building)
        cand = main[main["building"].map(lambda b: matches_building(b, bld))]
        rows.append(
            {
                "load_id": r.id,
                "label": getattr(r, "label", ""),
                "rated_kVA": r.rated_kVA,
                "register_building": bld,
                "candidate_meters": "; ".join(
                    f"{m.meter} ({m.place})" for m in cand.itertuples()
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("rated_kVA", ascending=False)
        .reset_index(drop=True)
    )
