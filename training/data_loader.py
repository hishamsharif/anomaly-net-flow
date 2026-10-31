"""
DuckDB-backed loader for the labelled_flows label store.

Training reads go through DuckDB for fast columnar scans.
DuckDB attaches to the SQLite DB file that the API writes to via aiosqlite.
"""
from __future__ import annotations
import duckdb
import pandas as pd

from api.config import settings


def _conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    if settings.database_url.startswith("sqlite"):
        db_path = settings.database_url.replace("sqlite+aiosqlite:///", "").lstrip("./")
        conn.execute(f"ATTACH '{db_path}' AS pg (TYPE sqlite)")
    return conn


_DAY_ORDER = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}


def _sort_csvs_by_day(paths: list[str]) -> list[str]:
    """
    Sort CIC-IDS2017 CSVs Monday→Friday so the 80/10/10 time-ordered split
    always puts Mon-Thu in training and Fri in val/test.
    Files without a recognisable day name sort last.
    """
    import os

    def key(p: str) -> tuple[int, str]:
        name = os.path.basename(p).lower()
        for day, rank in _DAY_ORDER.items():
            if day in name:
                return (rank, p)
        return (99, p)

    return sorted(paths, key=key)


def load_initial(bundled_dir: str) -> pd.DataFrame:
    """Load all CIC-IDS2017 CSV files in chronological (Mon→Fri) order."""
    import glob
    import os
    paths = glob.glob(os.path.join(bundled_dir, "**/*.csv"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {bundled_dir}")
    paths = _sort_csvs_by_day(paths)
    dfs = []
    conn = duckdb.connect(":memory:")
    for p in paths:
        dfs.append(conn.execute(f"SELECT * FROM read_csv_auto('{p}')").df())
    conn.close()
    return pd.concat(dfs, ignore_index=True)


def _split_group(path: str) -> str:
    """
    Assign a CIC-IDS2017 CSV to a split group.
    - Non-Friday files → 'train'
    - Friday morning   → 'val'
    - Friday afternoon → 'test'
    """
    import os
    name = os.path.basename(path).lower()
    if "friday" not in name:
        return "train"
    if "morning" in name:
        return "val"
    return "test"


def load_initial_by_day(
    bundled_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CIC-IDS2017 CSVs using a strict day-boundary split:
      train → Mon, Tue, Wed, Thu (all non-Friday files)
      val   → Friday WorkingHours Morning
      test  → Friday WorkingHours Afternoon (DDos + PortScan)

    Returns (train_df, val_df, test_df).
    """
    import glob
    import os

    paths = glob.glob(os.path.join(bundled_dir, "**/*.csv"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {bundled_dir}")

    groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for p in paths:
        groups[_split_group(p)].append(p)

    if not groups["val"]:
        raise FileNotFoundError(
            f"No Friday-Morning CSV found in {bundled_dir}. "
            "Expected a file whose name contains both 'Friday' and 'Morning'."
        )
    if not groups["test"]:
        raise FileNotFoundError(
            f"No Friday-Afternoon CSV found in {bundled_dir}. "
            "Expected files whose names contain 'Friday' but not 'Morning'."
        )

    conn = duckdb.connect(":memory:")

    def _load_group(file_paths: list[str]) -> pd.DataFrame:
        sorted_paths = _sort_csvs_by_day(file_paths)
        dfs = [conn.execute(f"SELECT * FROM read_csv_auto('{p}')").df()
               for p in sorted_paths]
        return pd.concat(dfs, ignore_index=True)

    train_df = _load_group(groups["train"])
    val_df   = _load_group(groups["val"])
    test_df  = _load_group(groups["test"])
    conn.close()
    return train_df, val_df, test_df


def load_cumulative(max_flow_id: int) -> pd.DataFrame:
    """XGBoost: all labelled_flows rows up to the savepoint."""
    conn = _conn()
    df = conn.execute(
        "SELECT * FROM pg.labelled_flows WHERE id <= ?", [max_flow_id]
    ).df()
    conn.close()
    return df


def load_incremental(prev_max_id: int, new_max_id: int) -> pd.DataFrame:
    """AE: rows since the previous savepoint."""
    conn = _conn()
    df = conn.execute(
        "SELECT * FROM pg.labelled_flows WHERE id > ? AND id <= ?",
        [prev_max_id, new_max_id],
    ).df()
    conn.close()
    return df


def load_replay_buffer(max_flow_id: int, fraction: float) -> pd.DataFrame:
    """Stratified sample from pre-savepoint rows for AE replay buffer."""
    conn = _conn()
    df = conn.execute(
        """
        SELECT * FROM pg.labelled_flows
        WHERE id <= ?
        USING SAMPLE ? PERCENT (stratify ON label)
        """,
        [max_flow_id, fraction * 100],
    ).df()
    conn.close()
    return df


def get_max_flow_id() -> int:
    conn = _conn()
    result = conn.execute("SELECT COALESCE(MAX(id), 0) FROM pg.labelled_flows").fetchone()
    conn.close()
    return int(result[0])
