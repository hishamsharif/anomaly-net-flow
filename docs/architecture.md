# AnomalyNetFlow — Solution Architecture

> Living document. Source of truth: Claude Docs https://claude.ai/artifact/H1ZQBwatmiRrM8a6ABNwKk

---

## 1. Purpose & Scope

**AnomalyNetFlow** is a network intrusion detection system (A2 individual coursework) that directly implements the layered design recommended by the **A1 group report** (Pattern Recognition for network IDS — `dev/log-analytics/`).

**A1 findings that drove A2 design:**

| A1 finding | A2 design response |
|---|---|
| RF ≈ perfect F1 on known attacks; recall collapses on unseen attack types (day-based split §4.5) | Stage 2 — Known-Attack Classifier catches known threats; Stage 3 catches what it misses |
| AE generalises to novel/zero-day attacks; AUC holds under day-based split | Stage 3 — Anomaly Detector (MLP-AE) trained on benign only; always runs after Stage 2 |
| Both methods complementary; neither alone is sufficient (§4.6) | 3-stage pipeline: Rule Engine → Classifier → Detector — all three always run |
| "Model retrained based on analyst feedback" stated in pipeline §3.1 but not implemented | Incremental fine-tune loop: 50 analyst verdicts → unlock Fine-tune |
| p95 AE threshold (DECISIONS.md §D5) | Same p95 threshold — same rationale, same calibration method |

AnomalyNetFlow adds a **training-first UI** that gives a SOC analyst full control over when models train, and a persistent **human-in-the-loop label store** that turns analyst verdicts into training data.

**In scope:** CIC-IDS2017 as the initial labelled dataset; CSV (CIC-IDS2017 format) and PCAP as inference inputs; XGBoost + MLP-AE as v1 models; human-in-the-loop labelling via analyst feedback; incremental fine-tuning from the label store.

**Out of scope:** Live packet capture; real-time streaming inference (planned M5).

---

## 2. Design Principles

- **Single source of truth** — feature registry at `GET /api/features`; no hardcoded column lists in JS or Python.
- **Durable training** — In-process `ThreadPoolExecutor` with SSE progress streaming; polling fallback (`GET /api/training/status/{run_id}`) keeps the UI updated even if the SSE connection drops.
- **No hard-coded feature lists** — UI reads from the feature registry endpoint.
- **Minimal runtime dependencies** — SQLite (file-based), no Redis, no Docker required to run locally.
- **Incremental by default** — v1 ships with the label store and fine-tune loop; swapping models is one registry entry.

---

## 3. System Overview

```
┌─────────────┐   CSV / PCAP upload      ┌──────────────────────────────────┐
│  React UI   │ ─────────────────────▶  │  FastAPI  (port 8000)            │
│  (Vite 5)   │ ◀─ SSE progress ──────  │  /api/predict/batch              │
│  3 tabs:    │                          │  /api/training/start             │
│  About      │   analyst feedback       │  /api/alerts/{id}/feedback       │
│  Training   │ ─────────────────────▶  │  /api/models/active              │
│  Inference  │                          │  /api/features                   │
└─────────────┘                          └───────────┬──────────────────────┘
                                                     │ background thread
                                         ┌───────────▼──────────────────────┐
                                         │  ThreadPoolExecutor (in-process) │
                                         │  run_initial_training()          │
                                         │  run_fine_tune()                 │
                                         │  SSE progress via asyncio.Queue  │
                                         └───────────┬──────────────────────┘
                                                     │ read/write
                                         ┌───────────▼──────────────────────┐
                                         │  SQLite (file-based, no server)  │
                                         │  labelled_flows (analyst verdicts)│
                                         │  training_savepoints             │
                                         │  model_versions, alerts, …       │
                                         │                                  │
                                         │  DuckDB (in-process)             │
                                         │  fast columnar reads at train    │
                                         └──────────────────────────────────┘
```

---

## 4. Technology Stack

### Frontend
| | |
|---|---|
| Framework | React 18 + TypeScript + Vite 5 |
| Routing | react-router-dom v6 (3 tabs: About / Training / Inference) |
| API communication | Native `fetch()` + `EventSource` (SSE) for live training progress |
| State | React `useState` / `useEffect`; polling fallback when SSE disconnects |

### Backend
| | |
|---|---|
| API | FastAPI + Uvicorn |
| ORM | SQLAlchemy 2 async |
| Task queue | `ThreadPoolExecutor` (in-process) — no Redis/Celery needed |
| DB | SQLite via aiosqlite (file-based, zero setup) |
| Analytical reads | DuckDB in-process (attaches SQLite; reads CIC-IDS2017 CSVs directly) |

### ML
| Component | v1 | Upgrade path |
|---|---|---|
| Supervised classifier | **XGBoost** — full cumulative retrain; native SHAP. **Random Forest** — bootstrap aggregation; balanced class weight; SHAP TreeExplainer | **FT-Transformer** — weight-based fine-tuning; shared encoder |
| Anomaly detector | **MLP Autoencoder** (fully-connected, MSE loss) + **Isolation Forest** (sklearn, no GPU) | **Transformer-AE / VAE** — self-attention over feature tokens |
| Class imbalance | SMOTE on train split only | — |
| Explainability (classifier) | SHAP TreeExplainer (XGBoost + Isolation Forest) | Attention weights |
| Explainability (MLP-AE) | Per-feature reconstruction error `(x − x̂)²` — no SHAP needed | — |
| Metrics | F1 macro, PR-AUC, Recall, FAR, AMR | — |
| Training reads | DuckDB in-process | — |
| Component registry | `core/registry.py` — add class + one line | — |
| Fine-tune XGBoost | Full cumulative retrain (no weight carry-over) | Warm-start trees at > 10 M rows |
| Fine-tune AE | Load weights → freeze encoder → fine-tune + replay | Shared encoder joint fine-tuning |
| Fine-tune IF | Full retrain on cumulative benign flows (no weight carry-over) | — |

---

## 5. ML Architecture

### 3-Stage Inference Pipeline

```
Flow features (72 cols)
        ↓
┌───────────────────┐
│  Rule Engine      │  10 deterministic rules  →  BENIGN (fast-path exit)
│  (Stage 1)        │
└────────┬──────────┘
         │ SUSPICIOUS
┌────────▼──────────┐
│  Known-Attack     │  Supervised classifier   →  predicted label + confidence
│  Classifier       │  15 CIC-IDS2017 classes
│  (Stage 2)        │  XGBoost or Random Forest
└────────┬──────────┘
         ↓
┌────────▼──────────┐
│  Anomaly Detector │  Anomaly score (MSE)     →  above p95 threshold = anomaly
│  (Stage 3)        │  MLP-AE or Isolation Forest
└────────┬──────────┘
         ↓
  Consensus → SAFE / SUSPICIOUS / CONFIRMED_CRITICAL
```

### Why XGBoost + Random Forest
Kotwal & Lanjekar (2024): XGBoost F1 = 95.95% vs RF 88.7% on CIC-IDS2017. XGBoost: native SHAP, handles class imbalance with `scale_pos_weight`, fast retrain (~60s on 2.8M rows). RF: lower hyperparameter sensitivity, built-in bootstrap aggregation reduces overfitting on small incremental datasets, `class_weight="balanced"` handles imbalance without resampling overhead. Both use full cumulative retrain (no weight carry-over) and share the same `predict() → (label, confidence)` interface.

### Why MLP Autoencoder (v1)
Multi-Layer Perceptron encoder-decoder, MSE reconstruction loss. Correct inductive bias for tabular network flow data — no convolutions (for images), no recurrence (for sequences). Fast to train and debug. Interpretable threshold (p95 on benign validation). Per-feature reconstruction error `(x − x̂)²` gives free explainability with no SHAP overhead. Upgrade to Transformer-AE when self-attention over feature tokens is needed.

### Anomaly Threshold Selection

The anomaly detector outputs a continuous reconstruction error score (mean MSE across all features). A binary decision (normal / anomaly) requires a threshold. AnomalyNetFlow uses a **percentile-based threshold** computed on benign validation data:

```python
threshold = np.percentile(reconstruction_errors(X_benign_val), 95)
```

**Why p95?**  
Setting the threshold at the 95th percentile of benign errors means exactly 5% of benign validation flows will exceed it — this is the **False Alarm Rate (FAR) floor by construction**. It is a deliberate tradeoff:

| Percentile | FAR floor | Detection sensitivity |
|---|---|---|
| p90 | 10% | Higher — catches more subtle deviations |
| p95 | 5% | Balanced — default |
| p99 | 1% | Lower — fewer false positives, more misses |

**Tradeoff:** Lowering the threshold (p90) increases recall but drowns analysts in false positives; raising it (p99) reduces alert fatigue but risks missing low-and-slow attacks.

**Saved artefact:** The threshold is serialised inside `detector.joblib` so inference uses exactly the same cutoff as training.

**Future — threshold sweep:** A `training/search/threshold_sweep.py` (Phase 2) will iterate over p80–p99 and plot the ROC / precision-recall curve, letting the operator choose the percentile that matches their acceptable FAR for the deployment environment.

---

### Transfer learning path (AE)
NSL-KDD (39 features) pretrain → CIC-IDS2017 (72 features) fine-tune. Freeze encoder first N epochs, then unfreeze. FT-Transformer upgrade: shares encoder with AE for a joint supervised + unsupervised backbone.

---

## 6. Data Architecture

### Label Store

Single growing table — CIC-IDS2017 bulk-loaded at startup; analyst verdicts appended on every review.

```sql
CREATE TABLE labelled_flows (
    id               BIGSERIAL PRIMARY KEY,
    features_json    JSONB NOT NULL,
    label            TEXT NOT NULL,          -- BENIGN | DoS | PortScan | …
    source           TEXT NOT NULL,          -- cic_ids2017 | analyst_confirmed | analyst_override
    predicted_label  TEXT,
    model_version_id INT REFERENCES model_versions(id),
    analyst_id       TEXT,
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE training_savepoints (
    id               SERIAL PRIMARY KEY,
    model_version_id INT REFERENCES model_versions(id),
    max_flow_id      BIGINT NOT NULL,        -- labelled_flows.id at training-start
    row_count        INT NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT now()
);
```

**XGBoost fine-tune:** `SELECT * FROM labelled_flows WHERE id <= new_savepoint.max_flow_id` (all data to date, cumulative retrain)

**AE fine-tune:** `prev_savepoint < id <= new_savepoint` + 5–10% stratified replay buffer (prevents catastrophic forgetting)

### Inference input formats

| Format | Adapter | Labels? |
|---|---|---|
| CSV (CIC-IDS2017 style) | `CSVInferenceAdapter` — drops Label col silently | No — inference only |
| PCAP / PCAPNG | `PcapAdapter` — NFStream bidirectional flow extraction (CICFlowMeter-compatible features) | No — inference only |

Both feed the inference pipeline only. Analyst verdicts on the resulting alerts become the labelled increment for the next fine-tune.

---

## 7. Training Lifecycle

```
App starts → GET /api/models/active
    ├── null  → "Train on CIC-IDS2017" button
    └── model → stats card + confusion matrix + "Fine-tune" button (unlocks at N ≥ K verdicts)

Initial training (POST /api/training/start mode=initial):
  ThreadPoolExecutor thread:
    DuckDB reads CIC-IDS2017 CSVs → FeatureEngineeringPipeline (VT → corr → MI → per-path scaler)
    → day-based split (Mon–Thu train / Fri val / Fri-afternoon test)
    → SMOTE on train only → XGBoost full train + AE fit on benign train
    → artefacts saved to models/
    → SSE events streamed to UI via asyncio.Queue

Fine-tune (POST /api/training/start mode=fine_tune):
  ThreadPoolExecutor thread:
    XGBoost: full retrain on cumulative labelled_flows (id ≤ new savepoint)
    AE:      incremental rows + 5–10% replay → load prev weights → freeze encoder → fine-tune
    → SSE events streamed to UI
```

**Day-based split** (mandatory, not random — per A1 Group report §4.4): time-ordered, tests generalisation to unseen attack variants on Friday captures.

**K threshold** (default 50, configurable in `config.py`): minimum analyst verdicts since last training before Fine-tune unlocks.

---

## 8. Inference & Feedback Loop

Analyst uploads CSV or PCAP → batch inference → alerts table populated with predicted label per flow → analyst reviews → submits verdict (`analyst_confirmed` | `analyst_override`) + correct label → row appended to `labelled_flows` → verdict count increments toward K.

**Both confirmed and overridden verdicts are stored.** Only storing overrides would bias the fine-tune set entirely toward failure cases.

---

## 9. Feature Roadmap

| Milestone | Increment |
|---|---|
| M1 Core | XGBoost + Vanilla AE, label store, analyst feedback, fine-tune loop |
| M2 Explainability | SHAP waterfall charts in alert detail panel |
| M3 Multi-dataset | NSL-KDD + UNSW-NB15 tabs; dataset-specific feature registries |
| M4 Auth | JWT + analyst vs admin roles |
| M5 Export | PDF/CSV report generation |
| M6 Streaming | WebSocket inference; sub-100 ms per-flow latency |
| M7 Packet-based DL | 1D-CNN / LSTM on raw packet bytes — flags active connections before the flow completes (no 15 s wait for statistical feature aggregation); enables sub-second detection of zero-days and encrypted threats. Different input format from v1 flow features; requires a separate adapter and model stack alongside the flow-based pipeline. |

---

## 10. Repository Layout

```
anomaly-net-flow/
├── api/                           # FastAPI HTTP layer (no ML logic)
│   ├── main.py                    # Entry point: uvicorn api.main:app
│   ├── config.py                  # pydantic-settings (K threshold, paths, feature-eng knobs)
│   ├── database.py                # SQLAlchemy async + DuckDB helper
│   ├── db/models.py               # ORM: labelled_flows, savepoints, alerts, …
│   ├── tasks.py                   # In-process background training (ThreadPoolExecutor + SSE)
│   └── routers/                   # predict, training, models, alerts, features
├── core/                          # Shared model classes — zero HTTP dependency
│   ├── base.py                    # PreprocessingSpec · BaseClassifier · BaseAnomalyDetector protocols
│   ├── registry.py                # CLASSIFIERS + ANOMALY_DETECTORS dicts — add one line here
│   ├── store.py                   # ModelStore singleton (dual clf_scaler / det_scaler)
│   ├── classifiers/               # xgboost_clf.py · random_forest_clf.py
│   └── detectors/                 # mlp_ae.py · isolation_forest.py
├── training/                      # Batch training processes
│   ├── pipelines/
│   │   ├── initial.py             # CIC-IDS2017 initial training
│   │   └── fine_tune.py           # Incremental fine-tune from label store
│   ├── feature_eng/
│   │   └── pipeline.py            # FeatureEngineeringPipeline (VT → corr → MI → scaler)
│   └── data_loader.py             # DuckDB queries for labelled_flows
├── inference/                     # Runtime inference adapters + explainers
│   ├── adapters/
│   │   ├── csv_adapter.py         # CSV → feature DataFrame (inference only)
│   │   └── pcap_adapter.py        # PCAP → NFStream flow extraction → feature DataFrame
│   └── explainers/                # Milestone 2: SHAP + AE per-feature reconstruction error
├── ui/src/
│   ├── App.tsx                    # 3-tab layout + active model check
│   └── pages/                     # AboutPage · TrainingPage · InferencePage
├── data/CIC-IDS2017/              # Bundled dataset (.gitignored for size)
├── models/                        # Saved artefacts (.gitignored)
├── docs/                          # Architecture and design documents
├── tests/
├── ops.sh                         # start | stop | restart | status | logs | ui | train
└── requirements.txt
```

---

## Confirmed Decisions

- **Day-based split** is the mandatory default (Mon–Thu train / Fri val / Fri-afternoon test). CSVs are loaded in Mon→Fri alphabetical day-name order before the 80/10/10 row split.
- **Detector scaler = QuantileTransformer** (not RobustScaler). CIC-IDS2017 sparse features (e.g. `Flow Bytes/s`) have IQR≈0; RobustScaler sets `scale_=1` so outliers reach the AE unscaled, producing threshold of 38M. QuantileTransformer maps any distribution to [0,1] — threshold is ~0.01–0.1.
- **SMOTE cap = 50K per minority class** (configurable `SMOTE_MAX_PER_CLASS`). Default SMOTE makes all classes equal to the BENIGN majority (~1.5M rows) — 10 classes × 1.5M = 17.4M synthetic samples, 8+ hour training. Cap at 50K → ~2M total → 5–15 min training.
- **v1 models:** XGBoost or Random Forest (supervised) + MLP Autoencoder (anomaly). FT-Transformer + Transformer-AE as upgrade path.
- **XGBoost fine-tuning** is always full cumulative retrain — trees are additive, not weights.
- **AE fine-tuning** uses weight-based incremental fine-tune: load previous weights → freeze encoder → unfreeze.
- **Both analyst_confirmed and analyst_override** are appended to `labelled_flows`.
- **PCAP/CSV uploads are inference-only** — they never touch the training store directly.
- **SQLite + DuckDB**: SQLite (file-based, no server) for operational writes (analyst verdicts, model versions, alerts); DuckDB in-process attaches SQLite and reads CIC-IDS2017 CSV files directly at training time — no import, no database server needed.
- **GPU/CUDA**: MLP-AE auto-detects CUDA (`torch.cuda.is_available()`). State dicts saved on CPU for device-portable joblib files. XGBoost uses `tree_method="hist"` (change to `device="cuda"` for GPU-accelerated trees).
