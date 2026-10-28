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


def load_initial(bundled_dir: str) -> pd.DataFrame:
    """Load all CIC-IDS2017 CSV files from the bundled dataset directory."""
    import glob
    import os
    paths = glob.glob(os.path.join(bundled_dir, "**/*.csv"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {bundled_dir}")
    conn = duckdb.connect(":memory:")
    df = conn.execute(
        f"SELECT * FROM read_csv_auto({paths!r}, union_by_name=True)"
    ).df()
    conn.close()
    return df


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
