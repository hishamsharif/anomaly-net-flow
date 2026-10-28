from __future__ import annotations
from typing import Protocol
import pandas as pd


class DataAdapter(Protocol):
    """
    Convert an uploaded file into a feature DataFrame ready for inference.
    Adapters are inference-only — they never write to the label store.
    The label column is dropped silently if present (it is not a training input).
    """

    def load(self, path: str) -> pd.DataFrame:
        """
        Returns a DataFrame where each row is one network flow.
        Columns match the 72-feature model schema.
        Missing optional columns are left as NaN; the preprocessor imputes them.
        Label / metadata columns (if present) are dropped.
        """
        ...
