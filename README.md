# AnomalyNetFlow

Incremental network intrusion detection with human-in-the-loop labelling.  
XGBoost · Random Forest · MLP Autoencoder · Isolation Forest · CIC-IDS2017

Architecture: [`docs/architecture.md`](docs/architecture.md)

---

## Prerequisites

| # | What | Where to get it |
|---|---|---|
| 1 | **Miniconda** (or Anaconda) | https://docs.conda.io/en/latest/miniconda.html |
| 2 | **Node.js ≥ 20** | https://nodejs.org |
| 3 | **CIC-IDS2017 CSV files** in `data/CIC-IDS2017/` | https://www.unb.ca/cic/datasets/ids-2017.html |

No Redis. No Docker. No database server.  
SQLite is file-based and created automatically on first run.

---

## Quick start

### 1 — Run setup (once)

```bash
bash setup.sh
```

Creates the `dl_env` conda environment, installs Python + UI dependencies, copies `.env.demo` → `.env`, and checks for the dataset.

### 2 — Start the API

```bash
bash ops.sh start
```

API: **http://localhost:8000** · Swagger docs: **http://localhost:8000/docs**

### 3 — Start the UI *(separate terminal)*

```bash
bash ops.sh ui
```

UI: **http://localhost:5173**

---

## First run: initial training

Open the **Train & Evaluate** tab — a *"Train on CIC-IDS2017"* button appears because no model is active yet. Select your classifier and detector, then click it, or trigger from the terminal:

```bash
bash ops.sh train initial
```

Training reads the CIC-IDS2017 CSVs directly (no import step), runs feature engineering, trains the selected Classifier + Anomaly Detector, and streams live progress via SSE to the Training tab.

**Available models (selectable in the UI):**

| Role | Options |
|---|---|
| Classifier | `xgboost` (default), `random_forest` |
| Anomaly Detector | `mlp_ae` (default), `isolation_forest` |

---

## Operations reference

```
bash setup.sh                  First-time setup (run once)

bash ops.sh start              Start the API in background (logs → logs/api.log)
bash ops.sh fg                 Start the API in foreground (logs direct to terminal, Ctrl-C to stop)
bash ops.sh stop               Stop the API
bash ops.sh restart            Restart the API
bash ops.sh status             Show running status + health check
bash ops.sh logs               Tail the background log  (Ctrl-C to exit)
bash ops.sh ui                 Start the React UI  (port 5173)
bash ops.sh train              Trigger initial training
bash ops.sh train fine_tune    Trigger incremental fine-tune
bash ops.sh verify             Check setup is complete
bash ops.sh open               Open the UI in your browser
bash ops.sh help               Show all commands
```

**Logging:**

| Mode | How to run | Where logs go |
|---|---|---|
| Background (default) | `bash ops.sh start` | `logs/api.log` — read with `bash ops.sh logs` |
| Foreground (troubleshooting) | `bash ops.sh fg` | Directly to terminal — Ctrl-C to stop |

---

## How it works

```
data/CIC-IDS2017/*.csv         Original dataset — stays as CSV, no import
         │
         │   DuckDB reads directly at training time
         ▼
  Initial model trained (Classifier + Anomaly Detector)
         │
  Analyst uploads CSV or PCAP → 3-stage inference pipeline
  (Rule engine → Classifier → MLP Autoencoder)
         │
  Predicted label + anomaly score per flow, displayed in UI
         │
  Analyst confirms or corrects labels → SQLite
         │
  anomaly_net_flow.db          SQLite (analyst verdicts only)
         │
         │   Unlocks after 50 verdicts (configurable)
         ▼
  Incremental fine-tune → improved model version
```

---

## 3-Stage Inference Pipeline

| Stage | Component | What it does |
|---|---|---|
| 1 | **Rule Engine** | Fast deterministic rules — flags obviously benign traffic and exits early |
| 2 | **Classifier** | XGBoost or Random Forest — predicts one of 15 attack classes or BENIGN |
| 3 | **Anomaly Detector** | MLP Autoencoder or Isolation Forest — catches novel/zero-day traffic the classifier hasn't seen |

Every alert includes an explainability signal:
- Classifier → SHAP per-feature contribution
- MLP Autoencoder → per-feature reconstruction error `(x − x̂)²` (no SHAP needed)

---

## Human-in-the-loop feedback

1. Upload CSV or PCAP in the **Inference** tab
2. Review flagged flows — each shows predicted label and anomaly score
3. Submit a verdict:
   - **Confirm** — prediction was correct
   - **Override** — select the correct label
4. Both verdict types are stored in the label store
5. After 50 verdicts, **Fine-tune** unlocks on the Training tab

---

## API endpoints

Base URL: `http://localhost:8000` · Full interactive docs: `http://localhost:8000/docs`

| Method | Path | Description | Requires trained model? |
|---|---|---|---|
| GET | `/health` | Health check | No |
| GET | `/api/models/active` | Active model or `null` | No |
| GET | `/api/models/` | All model versions + metrics | No |
| POST | `/api/models/{id}/promote` | Promote a version to active | No |
| POST | `/api/training/start` | Start training — returns `run_id` | No |
| GET | `/api/training/events/{run_id}` | SSE stream of live training progress | No |
| GET | `/api/training/available-types` | Classifier + detector options | No |
| GET | `/api/features/` | Feature list from the active model | Yes |
| POST | `/api/predict/batch` | Upload CSV or PCAP for batch inference | Yes |
| GET | `/api/alerts/` | Inference alerts (paginated) | Yes |
| POST | `/api/alerts/{id}/feedback` | Submit analyst verdict | Yes |
| GET | `/api/alerts/metrics` | Verdict count toward fine-tune threshold | Yes |

---

## Project structure

```
anomaly-net-flow/
├── setup.sh                      First-time setup
├── ops.sh                        All runtime operations
├── api/                          FastAPI HTTP layer (no ML logic)
│   ├── main.py                   Entry point: uvicorn api.main:app
│   ├── config.py                 Settings (.env / environment variables)
│   ├── database.py               SQLAlchemy async (SQLite) + DuckDB helper
│   ├── tasks.py                  In-process background training (ThreadPoolExecutor + SSE)
│   ├── db/
│   │   ├── models.py             ORM: labelled_flows, alerts, model_versions, …
│   │   └── crud.py               Sync DB helpers for background training threads
│   └── routers/                  predict, training, models, alerts, features
├── core/                         Shared model classes — zero HTTP dependency
│   ├── base.py                   PreprocessingSpec · BaseClassifier · BaseAnomalyDetector
│   ├── registry.py               Add one line here to expose a new model in the UI
│   ├── store.py                  ModelStore singleton (dual clf_scaler / det_scaler)
│   ├── classifiers/              xgboost_clf.py · random_forest_clf.py
│   └── detectors/                mlp_ae.py · isolation_forest.py
├── training/                     Batch training processes
│   ├── pipelines/
│   │   ├── initial.py            CIC-IDS2017 CSVs → Classifier + Detector
│   │   └── fine_tune.py          Analyst verdicts from SQLite → incremental update
│   ├── feature_eng/
│   │   └── pipeline.py           FeatureEngineeringPipeline (VT → corr → MI → scaler)
│   └── data_loader.py            DuckDB queries (CSV + SQLite)
├── inference/                    Runtime inference adapters
│   └── adapters/
│       ├── csv_adapter.py        CSV upload → feature DataFrame
│       └── pcap_adapter.py       PCAP upload → NFStream flow extraction
├── ui/                           React 18 + TypeScript + Vite 5
│   └── src/pages/                AboutPage · TrainingPage · InferencePage
├── data/CIC-IDS2017/             Place dataset CSV files here (.gitignored)
├── models/                       Saved model artefacts  (created at runtime, .gitignored)
├── docs/architecture.md          Full system architecture and design decisions
├── requirements.txt
└── .env.demo                     Copy to .env to override defaults
```

---

## Configuration

All settings have working defaults — no configuration needed for a local run.

```bash
cp .env.demo .env   # then edit as needed
```

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./anomaly_net_flow.db` | SQLite database path |
| `BUNDLED_DATASET_DIR` | `data/CIC-IDS2017` | CIC-IDS2017 CSV folder |
| `FINE_TUNE_K_THRESHOLD` | `50` | Analyst verdicts needed to unlock fine-tune |
| `AE_REPLAY_BUFFER_FRACTION` | `0.1` | Old benign data replayed during AE fine-tune (prevents forgetting) |
| `AE_ANOMALY_THRESHOLD_PERCENTILE` | `95` | p-value for anomaly threshold (p90=more alerts, p99=fewer) |
| `MAX_UPLOAD_MB` | `200` | Max inference file upload size |

---

## Extending with a new model

1. Create `core/classifiers/my_model.py` implementing `BaseClassifier` (see `core/base.py`)
   - Declare `preprocessing = PreprocessingSpec(needs_scaling=..., preferred_scalers=(...))` as a class variable
2. Add one line to `core/registry.py`:
   ```python
   CLASSIFIERS["my_model"] = MyModelClassifier
   ```
3. The model appears automatically in the Training tab dropdown and `GET /api/training/available-types`

The same pattern applies to anomaly detectors via `ANOMALY_DETECTORS` and `core/detectors/`.
