"""Read / write the RL problem as a workbook (``rl.xlsx``), the case study's input form.

Sheets
* ``problem``      ``key | value | source``; dotted keys address the groups, e.g.
  ``scenario.added_pv_kwp``, ``episode.episode_days``, ``events.source``, ``agent.gamma``;
  lists are JSON (``[128, 128]``), an empty value means ``None``
* ``reward``       ``metric | weight | transform | threshold | params | source`` (params JSON)
* ``observation``  ``name | params | source``
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.operation.rl.problem import RLProblem

DATA_DIR = Path(__file__).resolve().parent / "data"
GROUPS = ("scenario", "episode", "events", "action", "agent")


def _parse(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        if s.startswith(("[", "{")):
            return json.loads(s)
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _fmt(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, (list, dict)) else value


def problem_tables(
    problem: RLProblem, source: str = "default"
) -> dict[str, pd.DataFrame]:
    """The three sheets of a problem."""
    cfg = asdict(problem)
    rows = []
    for k in (
        "id",
        "power_flow_mode",
        "battery_policy",
        "forecast_mode",
        "reward_baseline",
        "reward_scale",
    ):
        rows.append((k, cfg[k], source))
    for g in GROUPS:
        for k, v in cfg[g].items():
            rows.append((f"{g}.{k}", _fmt(v), source))
    return {
        "problem": pd.DataFrame(rows, columns=["key", "value", "source"]),
        "reward": pd.DataFrame(
            [
                (
                    t["metric"],
                    t["weight"],
                    t["transform"],
                    t["threshold"],
                    _fmt(t["params"]),
                    source,
                )
                for t in cfg["reward"]
            ],
            columns=["metric", "weight", "transform", "threshold", "params", "source"],
        ),
        "observation": pd.DataFrame(
            [(t["name"], _fmt(t["params"]), source) for t in cfg["observation"]],
            columns=["name", "params", "source"],
        ),
    }


def write_problem(problem: RLProblem, path: Path, source: str = "default") -> Path:
    """Write ``problem`` as workbook."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as xw:
        for name, df in problem_tables(problem, source).items():
            df.to_excel(xw, sheet_name=name, index=False)
    return path


def load_problem(path: Path = DATA_DIR / "rl.xlsx") -> RLProblem:
    """Read a workbook into an ``RLProblem`` (validated)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RL workbook not found: {path}")
    sheets = pd.read_excel(
        path, sheet_name=["problem", "reward", "observation"], dtype=object
    )
    cfg: dict[str, Any] = {g: {} for g in GROUPS}
    for key, value in zip(sheets["problem"]["key"], sheets["problem"]["value"]):
        v = _parse(value)
        if "." in key:
            group, name = key.split(".", 1)
            if group not in cfg:
                raise ValueError(f"{path.name}: unknown group '{group}' in key '{key}'")
            cfg[group][name] = v
        else:
            cfg[key] = v
    r = sheets["reward"]
    cfg["reward"] = [
        dict(
            metric=row.metric,
            weight=float(row.weight),
            transform=row.transform,
            threshold=float(row.threshold),
            params=_parse(row.params) or {},
        )
        for row in r.itertuples()
    ]
    cfg["observation"] = [
        dict(name=row.name, params=_parse(row.params) or {})
        for row in sheets["observation"].itertuples()
    ]
    for g in GROUPS:
        if not cfg[g]:
            del cfg[g]
    if "scenario" in cfg:  # Scenario fields are numbers
        cfg["scenario"] = {k: v for k, v in cfg["scenario"].items() if v is not None}
    return RLProblem.from_dict(cfg)
