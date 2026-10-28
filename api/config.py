from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "sqlite+aiosqlite:///./anomaly_net_flow.db"

    # Training
    bundled_dataset_dir: str = "data/CIC-IDS2017"
    model_artefact_dir: str = "models"
    fine_tune_k_threshold: int = 50        # verdicts needed to unlock Fine-tune
    ae_anomaly_threshold_percentile: float = 95.0
    ae_replay_buffer_fraction: float = 0.10
    ae_freeze_epochs: int = 5

    # Feature engineering — tune these to trade feature count for performance
    # Step 1: drop near-constant features  (0 = constants only, 0.01, 0.05 progressively stricter)
    variance_threshold: float = 0.0
    # Step 2: drop one of each pair with |Pearson r| above this (redundancy pruning)
    correlation_threshold: float = 0.95
    # Step 3: keep only top-K features ranked by mutual information with the label
    #         0 = keep all that survived steps 1 & 2
    feature_selection_k: int = 0
    # Scaler applied after feature selection (robust | standard | quantile | maxabs | none)
    scaler_type: str = "robust"

    # Inference adapters
    max_upload_mb: int = 100


settings = Settings()
