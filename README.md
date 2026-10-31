# AnomalyNetFlow — CS7003NU A2 Individual Coursework

## The Problem

SOC analysts spend most of their shift on repetitive alert triage. Most alerts are known attack patterns that a classifier can handle automatically. Novel attacks require human judgment — but today that judgment is discarded after each review, so the system never learns.

Underneath this sits a deeper issue: **concept drift**. Attack toolkits evolve — adversaries change their packet timing, sizes, and flag patterns to evade known signatures. A model trained on last year's captures degrades on this year's traffic, not because the model is wrong but because the threat landscape has moved. Static NIDS deployments have no mechanism to close this gap without a full manual retrain.

**AnomalyNetFlow's dual goal:**
1. **Automate** routine SOC triage so analysts focus on genuine unknowns, not repetitive classification
2. **Adapt** continuously — every analyst verdict is a supervision signal; fine-tuning incorporates new ground truth as it is observed, so the model tracks the evolving threat landscape rather than freezing at its training date

Every analyst verdict (confirm or override) is stored. Once 50 verdicts accumulate the model can be fine-tuned — the classifier absorbs what the analyst team has seen and labelled. Over time, previously novel attack patterns graduate from "analyst review required" to "automated classification".

## How it works

**Pipeline:** Rule Engine → Known-Attack Classifier → Anomaly Detector

| Stage | Automates | Analyst interaction |
|---|---|---|
| Rule Engine | Obvious benign traffic (fast-path exit) | None |
| Known-Attack Classifier (XGBoost) | Repeat attack patterns from training + past fine-tunes | Verdict stored as future training signal |
| Anomaly Detector (MLP-AE) | Flags deviations from learnt benign baseline — potential zero-days | Review + label → unlocks next fine-tune |

**Dataset:** CIC-IDS2017 · 2.8 M flows · 15 attack classes — provides Day 0 knowledge before any analyst interaction  
**Models:** XGBoost (default) · Random Forest · MLP Autoencoder (default) · Isolation Forest

Full design and problem statement: [`docs/architecture.md §0`](docs/architecture.md)

---

**A1 connection (§4.6):** Random Forest achieves near-perfect F1 on *known* attacks but its recall collapses on unseen attack types under a day-based split; the Autoencoder generalises better to novel/zero-day traffic because it models normality, not attack signatures. A1 concluded the most effective design is *layered*. AnomalyNetFlow implements exactly that, and adds the incremental feedback loop that A1 §3.1 described ("model retrained based on analyst feedback") but did not implement.

---

## Quick start — Docker (recommended for evaluation)

Requires only Docker Desktop. All Python, Node, and library dependencies are inside the images.

### Prerequisites

| # | What | Where |
|---|---|---|
| 1 | **Docker Desktop** | https://docs.docker.com/get-docker/ |
| 2 | **CIC-IDS2017 CSV files** placed in `data/CIC-IDS2017/` | https://www.unb.ca/cic/datasets/ids-2017.html |

### Steps

```bash
# 1. Build and start both containers
docker compose up --build

# 2. Open the UI  (waits for the API healthcheck before serving)
open http://localhost:8080

# 3. First time: click "Train on CIC-IDS2017" in the Training tab
#    Live progress streams to the Training tab via SSE
```

**Ports**

| Service | URL |
|---|---|
| UI (nginx SPA) | http://localhost:8080 |
| API + Swagger docs | http://localhost:8000/docs |

**Volume mounts** (data persisted on your host, not inside the container):

```
./data/    → /app/data    (CIC-IDS2017 CSVs + SQLite DB)
./models/  → /app/models  (trained model artefacts — persisted across restarts)
```

---

## Quick start — Local (conda)

### Prerequisites

| # | What | Where to get it |
|---|---|---|
| 1 | **Miniconda** (or Anaconda) | https://docs.conda.io/en/latest/miniconda.html |
| 2 | **Node.js ≥ 20** | https://nodejs.org |
| 3 | **CIC-IDS2017 CSV files** in `data/CIC-IDS2017/` | https://www.unb.ca/cic/datasets/ids-2017.html |

SQLite is file-based and created automatically on first run. No Redis, no database server.

### Steps

```bash
# 1. First-time setup (creates conda env, installs deps, copies .env.demo → .env)
bash setup.sh

# 2. Start the API
bash ops.sh start
# API: http://localhost:8000  |  Swagger: http://localhost:8000/docs

# 3. Start the UI  (separate terminal)
bash ops.sh ui
# UI: http://localhost:5173
```

---

## First run: initial training

Open the **Training** tab — a *"Train on CIC-IDS2017"* button appears because no model is active yet. Select your classifier and detector, then click it, or trigger from the terminal:

```bash
bash ops.sh train initial
```

Training reads the CIC-IDS2017 CSVs directly (no import step), runs feature engineering, trains the selected Classifier + Anomaly Detector, and streams live progress to the Training tab.

**Available models (selectable in the UI):**

| Role | Options |
|---|---|
| Classifier | `xgboost` (default), `random_forest` |
| Anomaly Detector | `mlp_ae` (default), `isolation_forest`, `transfer_ae` (UNSW-NB15 pretrain → CIC fine-tune) |

---

## Operations reference

```
bash setup.sh                  First-time setup (run once)

bash ops.sh start              Start the API in background (logs → logs/api.log)
bash ops.sh fg                 Start the API in foreground (Ctrl-C to stop)
bash ops.sh stop               Stop the API
bash ops.sh restart            Restart the API
bash ops.sh status             Show running status + health check
bash ops.sh logs               Tail the background log  (Ctrl-C to exit)
bash ops.sh ui                 Start the React UI  (port 5173)
bash ops.sh train initial      Trigger initial training
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
| 2 | **Known-Attack Classifier** | XGBoost or Random Forest — predicts one of 15 attack classes or BENIGN (supervised, trained with labels) |
| 3 | **Anomaly Detector** | MLP Autoencoder or Isolation Forest — catches novel/zero-day traffic the classifier hasn't seen (unsupervised, trained on benign only) |

Every alert includes an explainability signal:
- Classifier → SHAP per-feature contribution
- MLP Autoencoder → per-feature reconstruction error `(x − x̂)²`

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
| GET | `/api/training/status/{run_id}` | Poll current progress % (SSE fallback) | No |
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
│   │   ├── models.py             ORM: labelled_flows, alerts, model_versions
│   │   └── crud.py               Sync DB helpers for background training threads
│   └── routers/                  predict, training, models, alerts, features
├── core/                         Shared model classes — zero HTTP dependency
│   ├── base.py                   PreprocessingSpec · BaseClassifier · BaseAnomalyDetector
│   ├── registry.py               Add one line here to expose a new model in the UI
│   ├── store.py                  ModelStore singleton (dual clf_scaler / det_scaler)
│   ├── classifiers/              xgboost_clf.py · random_forest_clf.py
│   └── detectors/                mlp_ae.py · isolation_forest.py · transfer_ae.py
├── training/                     Batch training processes
│   ├── pipelines/
│   │   ├── initial.py            CIC-IDS2017 CSVs → Classifier + Detector
│   │   └── fine_tune.py          Analyst verdicts from SQLite → incremental update
│   ├── feature_eng/
│   │   └── pipeline.py           FeatureEngineeringPipeline (VT → corr → MI|RF-RFE → scaler)
│   └── data_loader.py            DuckDB queries (CSV + SQLite)
├── inference/                    Runtime inference adapters + explainability
│   ├── adapters/
│   │   ├── csv_adapter.py        CSV upload → feature DataFrame
│   │   └── pcap_adapter.py       PCAP upload → NFStream flow extraction
│   └── explainers/               SHAP (Stage 2) + AE per-feature reconstruction error (Stage 3)
├── scripts/
│   └── tune_xgb.py               Standalone XGBoost hyperparameter search (RandomizedSearchCV)
├── ui/                           React 18 + TypeScript + Vite 5
│   └── src/pages/                AboutPage · TrainingPage · InferencePage
├── data/CIC-IDS2017/             Place dataset CSV files here (.gitignored)
├── models/                       Saved model artefacts  (created at runtime, .gitignored)
├── Dockerfile.api                API container
├── Dockerfile.ui                 UI container (nginx)
├── docker-compose.yml            Two-container Docker setup
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
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/anomaly_net_flow.db` | SQLite database path |
| `BUNDLED_DATASET_DIR` | `data/CIC-IDS2017` | CIC-IDS2017 CSV folder |
| `FINE_TUNE_K_THRESHOLD` | `50` | Analyst verdicts needed to unlock fine-tune |
| `AE_REPLAY_BUFFER_FRACTION` | `0.1` | Old benign data replayed during AE fine-tune (prevents forgetting) |
| `AE_ANOMALY_THRESHOLD_PERCENTILE` | `95` | Anomaly decision boundary percentile (p90=more alerts, p99=fewer) |
| `SMOTE_MAX_PER_CLASS` | `50000` | Max synthetic samples per minority class (caps training set size) |
| `FEATURE_SELECTION_K` | `0` (disabled) | Keep only top-K features after VT + corr pruning; 0 = keep all |
| `FEATURE_SELECTION_METHOD` | `mi` | Feature selection method: `mi` (mutual info, fast) · `rfe` (RF-RFE, interaction-aware) · `none` |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost,http://localhost:3000` | Allowed browser origins (comma-separated) |
| `MAX_UPLOAD_MB` | `100` | Max inference file upload size |

---

## Code navigation

| If you want to understand… | Start here |
|---|---|
| The 3-stage pipeline end-to-end | `docs/architecture.md` §5 ML Architecture |
| Model protocols (what every classifier/detector must implement) | `core/base.py` — `BaseClassifier`, `BaseAnomalyDetector` |
| How XGBoost / MLP-AE are implemented | `core/classifiers/xgboost_clf.py`, `core/detectors/mlp_ae.py` |
| TransferAutoencoder (UNSW-NB15 pretrain → CIC fine-tune) | `core/detectors/transfer_ae.py` |
| RF-RFE feature selection | `training/feature_eng/pipeline.py` — `FeatureEngineeringPipeline(feature_selection_method="rfe")` |
| Initial training pipeline (CIC-IDS2017 → artefacts) | `training/pipelines/initial.py` |
| Incremental fine-tune pipeline | `training/pipelines/fine_tune.py` |
| Feature engineering (VT → corr → MI|RFE → scaler) | `training/feature_eng/pipeline.py` |
| How inference runs at request time | `api/routers/predict.py` |
| Background training + SSE progress | `api/tasks.py` |
| DB schema (label store, alerts, model versions) | `api/db/models.py` |
| How to add a new model in one line | `core/registry.py` |

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
