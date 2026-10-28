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
