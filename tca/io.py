from __future__ import annotations

from pathlib import Path

import pandas as pd


def _drop_noise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the empty leading column and the (blocked) Data Export Restricted columns.

    EMSX grid exports carry a leading comma (an 'Unnamed' column) and several
    `Data Export Restricted` columns that come through entirely empty. pandas
    de-duplicates the repeated header into `Data Export Restricted.1` etc.
    """
    drop = [
        c
        for c in df.columns
        if str(c).strip().lower().startswith("unnamed")
        or str(c).strip().lower().startswith("data export restricted")
    ]
    return df.drop(columns=drop) if drop else df


def load_parent_orders(csv_path: str | Path) -> pd.DataFrame:
    """Load the parent-order export (the primary analysis unit).

    Accepts either the CSV export or the .xlsx (if openpyxl is installed). The
    full `td_data_full.xlsx` pull carries the previously-blocked ArrPx /
    IntervalVWAP benchmarks; convert it to CSV once if openpyxl is unavailable.
    """
    path = Path(csv_path)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    return _drop_noise_columns(df)


def load_child_orders(csv_path: str | Path) -> pd.DataFrame:
    """Load the child-route CSV (diagnosis / attribution only).

    The child export carries no parent order id, so it links to parents only by
    `Security` (and, loosely, time). It does, however, carry `Volatil 30D` and
    `Avg Vol 20D`, which the parent export omits.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    return _drop_noise_columns(df)
