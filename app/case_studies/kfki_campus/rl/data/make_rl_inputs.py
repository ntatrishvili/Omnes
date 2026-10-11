"""Write the default RL inputs: ``rl.xlsx`` and an EXAMPLE charging-session file.

    python -m app.case_studies.kfki_campus.rl.data.make_rl_inputs

The example session file is produced by the dummy event generator: it shows the format, it is
**not** measured data. Replace it with the real session file (see ``README_RL.md``).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.rl.problem_io import DATA_DIR, write_problem
from app.case_studies.kfki_campus.rl.sessions import write_sessions
from app.operation.rl.events import DummyChargingEventGenerator
from app.operation.rl.problem import EventSpec, RLProblem


def default_problem() -> RLProblem:
    """2009 kWp PV (9 existing + 2000 added), 1450 kWh NaS, 6 chargers, dummy event generator."""
    return RLProblem(
        id="kfki_ev_v1g",
        scenario=dict(added_pv_kwp=2000.0, battery_kwh=1450.0, n_chargers=6),
        events=EventSpec(source="generator", generator="dummy"),
    )


def main(out_dir: Path = DATA_DIR) -> None:
    out_dir = Path(out_dir)
    write_problem(
        default_problem(), out_dir / "rl.xlsx", source="default (dsl: RL block)"
    )
    d = load_campus_data()
    chargers = {c: 22.0 for c in d.ev_chargers["id"].head(6)}
    events = DummyChargingEventGenerator().sample(
        np.random.default_rng(0),
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-06-15"),
        chargers,
    )
    write_sessions(events, out_dir / "charging_sessions_example.csv")
    print(
        f"wrote {out_dir / 'rl.xlsx'} and {out_dir / 'charging_sessions_example.csv'}"
    )


if __name__ == "__main__":
    main()
