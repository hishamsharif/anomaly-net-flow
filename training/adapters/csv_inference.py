import pandas as pd

# Metadata/label columns always excluded from model features
_DROP_COLS = {"Label", "Flow ID", "Timestamp", "Source IP", "Destination IP",
              "label", "flow_id", "timestamp", "source_ip", "destination_ip"}


class CSVInferenceAdapter:
    """
    Load a CIC-IDS2017-format CSV for inference.
    Label column dropped silently if present — this is an inference adapter,
    not a training data source.
    """

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        drop = [c for c in df.columns if c in _DROP_COLS or c.strip() in _DROP_COLS]
        df = df.drop(columns=drop, errors="ignore")
        # Coerce to numeric; non-parseable cells become NaN (imputed later)
        df = df.apply(pd.to_numeric, errors="coerce")
        return df
