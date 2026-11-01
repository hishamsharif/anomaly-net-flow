from datetime import datetime, timezone
from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LabelledFlow(Base):
    """Label store — single growing table; CIC-IDS2017 rows + analyst verdicts."""
    __tablename__ = "labelled_flows"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    features_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    # cic_ids2017 | analyst_confirmed | analyst_override
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    predicted_label: Mapped[str | None] = mapped_column(String(64))
    model_version_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("model_versions.id"))
    analyst_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class TrainingSavepoint(Base):
    """Records max_flow_id at training-start for each model version."""
    __tablename__ = "training_savepoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_versions.id"))
    max_flow_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class FeatureSchemaVersion(Base):
    """
    Versioned feature contract — tracks which extractor and feature set produced
    each trained model. Append-only; never overwrite a released schema record.
    """
    __tablename__ = "feature_schema_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schema_version: Mapped[str] = mapped_column(String(64))        # e.g. cicids2017-flow-schema-v1
    extractor_name: Mapped[str] = mapped_column(String(64))        # cicflowmeter-python | nfstream
    extractor_version: Mapped[str | None] = mapped_column(String(32))
    selected_feature_count: Mapped[int | None] = mapped_column(Integer)
    absent_features_json: Mapped[list] = mapped_column(JSON, default=list)
    compatibility_notes: Mapped[str | None] = mapped_column(Text)
    artefact_path: Mapped[str | None] = mapped_column(Text)        # path to feature_schema.json
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str | None] = mapped_column(String(128))      # human label, set after ID is known
    classifier_type: Mapped[str] = mapped_column(String(64))   # xgboost | random_forest | …
    detector_type: Mapped[str] = mapped_column(String(64))     # mlp_ae | isolation_forest | …
    artefact_path: Mapped[str] = mapped_column(Text, default="")
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    # training | active | archived | failed
    status: Mapped[str] = mapped_column(String(16), default="training")
    savepoint_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("training_savepoints.id"))
    feature_schema_version_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("feature_schema_versions.id")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(256))
    mode: Mapped[str] = mapped_column(String(16))   # initial | fine_tune
    row_count: Mapped[int | None] = mapped_column(Integer)
    model_version_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("model_versions.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    flow_id: Mapped[str] = mapped_column(String(64))
    consensus: Mapped[str] = mapped_column(String(32))
    severity: Mapped[str] = mapped_column(String(16))
    rule_verdict: Mapped[str] = mapped_column(String(16))
    predicted_label: Mapped[str | None] = mapped_column(String(64))
    classifier_confidence: Mapped[float | None] = mapped_column(Float)
    ae_score: Mapped[float | None] = mapped_column(Float)
    ai_skipped: Mapped[bool] = mapped_column(default=False)
    # analyst_confirmed | analyst_override | analyst_uncertain | analyst_escalate | null
    # Only analyst_confirmed and analyst_override are retraining_eligible.
    # Predictions are append-only; analyst_verdict is the only field updated post-creation.
    analyst_verdict: Mapped[str | None] = mapped_column(String(32))
    # False for analyst_uncertain / analyst_escalate; True for confirmed / override.
    retraining_eligible: Mapped[bool] = mapped_column(default=False)
    analyst_label: Mapped[str | None] = mapped_column(String(64))
    labelled_flow_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("labelled_flows.id"))
    model_version_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("model_versions.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class FeatureRegistry(Base):
    __tablename__ = "feature_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_versions.id"))
    required_features: Mapped[list] = mapped_column(JSON)
    optional_features: Mapped[list] = mapped_column(JSON)
    medians_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
