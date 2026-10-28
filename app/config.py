from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "sqlite+aiosqlite:///./anomaly_net_flow.db"
    duckdb_path: str = ":memory:"  # in-process; set to a file path for persistence

    # Training
    bundled_dataset_dir: str = "data/CIC-IDS2017"
    model_artefact_dir: str = "models"
    fine_tune_k_threshold: int = 50        # verdicts needed to unlock Fine-tune
    ae_anomaly_threshold_percentile: float = 95.0
    ae_replay_buffer_fraction: float = 0.10
    ae_freeze_epochs: int = 5

    # Inference adapters
    max_upload_mb: int = 100


settings = Settings()
