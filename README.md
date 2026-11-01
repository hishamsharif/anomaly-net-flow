# AnomalyNetFlow — CS7003NU A2 Individual Coursework

## The Problem

SOC analysts spend most of their shift on repetitive alert triage. Most alerts are known attack patterns that a classifier can handle automatically. Novel attacks require human judgment — but today that judgment is discarded after each review, so the system never learns.

Underneath this sits a deeper issue: **concept drift**. Attack toolkits evolve — adversaries change their packet timing, sizes, and flag patterns to evade known signatures. A model trained on last year's captures degrades on this year's traffic, not because the model is wrong but because the threat landscape has moved. Static NIDS deployments have no mechanism to close this gap without a full manual retrain.

**AnomalyNetFlow's dual goal:**
1. **Automate** routine SOC triage so analysts focus on genuine unknowns, not repetitive classification
2. **Adapt** continuously — every analyst verdict is a supervision signal; periodic AE fine-tuning incorporates new ground truth as it is observed, so the model tracks the evolving threat landscape rather than freezing at its training date

---

## Quick Start — Docker (recommended)

**Requires only Docker Desktop installed on your machine. No Python, Node, or conda needed.**

```bash
./setup.sh
```

That single command:
1. Checks Docker is running
2. Copies `.env.demo` → `.env` (first time)
3. Builds both containers (API + UI)
4. Starts the application in the background
5. Waits for the health check and prints the URLs

**Access the application:**

| | URL |
|---|---|
| Web UI | http://localhost:8080 |
| REST API + Swagger | http://localhost:8000/docs |

**Dataset (required for initial training):**

The CIC-IDS2017 dataset is auto-downloaded on first setup if an internet connection is available.  
If not, place the CSV files manually in `data/CIC-IDS2017/` before clicking "Train on CIC-IDS2017".  
Source: https://www.unb.ca/cic/datasets/ids-2017.html

**Docker commands after setup:**

```bash
docker compose logs -f api    # stream API logs
docker compose stop           # stop containers
docker compose down           # stop + remove containers
docker compose up -d          # restart (no rebuild needed)
```

---

## Quick Start — Local (no Docker)

If Docker is not available, run:

```bash
./setup.sh --local
```

**Prerequisites before running:**

| | What | Where |
|---|---|---|
| 1 | **Miniconda** (or Anaconda) | https://docs.conda.io/en/latest/miniconda.html |
| 2 | **Node.js ≥ 20** | https://nodejs.org |
| 3 | **CIC-IDS2017 CSV files** in `data/CIC-IDS2017/` | https://www.unb.ca/cic/datasets/ids-2017.html |

After setup completes:

```bash
# Terminal 1 — API
bash ops.sh start
# API: http://localhost:8000  |  Swagger: http://localhost:8000/docs

# Terminal 2 — UI
bash ops.sh ui
# UI: http://localhost:5173
```

---

## How it works

**Pipeline:** Rule Engine → Known-Attack Classifier → Anomaly Detector

| Stage | Component | Automates | Analyst interaction |
|---|---|---|---|
| 1 | **Rule Engine** | Obvious benign traffic (fast-path exit) | None |
| 2 | **Known-Attack Classifier** | Repeat attack patterns (XGBoost, trained on CIC-IDS2017) | Verdict stored as future AE training signal |
| 3 | **Anomaly Detector** | Novel / zero-day traffic (MLP Autoencoder, trained on benign only) | Review + label → unlocks periodic AE retrain |

**Dataset:** CIC-IDS2017 · 2.8 M flows · 15 attack classes — provides Day 0 knowledge before any analyst interaction  
**Models:** XGBoost (default) · Random Forest · MLP Autoencoder (default) · Isolation Forest

Full design and problem statement: [`docs/architecture.md`](docs/architecture.md)

---

**A1 connection (§4.6):** Random Forest achieves near-perfect F1 on *known* attacks but its recall collapses on unseen attack types under a day-based split; the Autoencoder generalises better to novel/zero-day traffic because it models normality, not attack signatures. A1 concluded the most effective design is *layered*. AnomalyNetFlow implements exactly that, and adds the incremental feedback loop that A1 §3.1 described ("model retrained based on analyst feedback") but did not implement.

---

## First run: initial training

Open the **Training → Train & Evaluate** tab. Since no model is active yet, you will see a *"Train on CIC-IDS2017"* card. Select your classifier and detector and click the button, or trigger from the terminal:

```bash
bash ops.sh train initial
```

Training reads the CIC-IDS2017 CSVs directly (no import step), runs feature engineering, trains the selected Classifier + Anomaly Detector, and streams live progress to the Training tab.

**Available models (selectable in the UI):**

| Role | Options |
|---|---|
| Classifier | `xgboost` (default), `random_forest` |
| Anomaly Detector | `mlp_ae` (default), `isolation_forest` |

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
4. Both verdict types are stored and count toward the retraining threshold
5. After 50 confirmed/override verdicts, **Retrain AE** unlocks on the Training tab

The XGBoost classifier is **frozen** at its initial CIC-IDS2017 weights permanently.  
Only the MLP Autoencoder is periodically fine-tuned (transfer learning from the previous AE checkpoint) using analyst-labelled benign flows.

---

## Operations reference (local mode)

```
bash ops.sh start              Start the API in background (logs → logs/api.log)
bash ops.sh fg                 Start the API in foreground (Ctrl-C to stop)
bash ops.sh stop               Stop the API
bash ops.sh restart            Restart the API
bash ops.sh status             Show running status + health check
bash ops.sh logs               Tail the background log (Ctrl-C to exit)
bash ops.sh ui                 Start the React UI dev server (port 5173)
bash ops.sh train initial      Trigger initial training
bash ops.sh train ae_retrain   Trigger periodic AE retrain
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
| GET | `/api/alerts/metrics` | Verdict count toward retrain threshold | Yes |

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
| `FINE_TUNE_K_THRESHOLD` | `50` | Analyst verdicts needed to unlock AE retrain |
| `AE_REPLAY_BUFFER_FRACTION` | `0.1` | Old benign data replayed during AE retrain (prevents forgetting) |
| `AE_ANOMALY_THRESHOLD_PERCENTILE` | `95` | Anomaly decision boundary percentile (p90=more alerts, p99=fewer) |
| `SMOTE_MAX_PER_CLASS` | `50000` | Max synthetic samples per minority class (caps training set size) |
| `FEATURE_SELECTION_K` | `0` (disabled) | Keep only top-K features after VT + corr pruning; 0 = keep all |
| `FEATURE_SELECTION_METHOD` | `mi` | Feature selection: `mi` (mutual info) · `rfe` (RF-RFE) · `none` |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost,http://localhost:3000` | Allowed browser origins |
| `MAX_UPLOAD_MB` | `100` | Max inference file upload size |

---

## Project structure

```
anomaly-net-flow/
├── setup.sh                      First-time setup (Docker by default; --local for conda)
├── ops.sh                        All runtime operations (local mode)
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
│   │   └── fine_tune.py          AE-only periodic retrain (analyst verdicts → AE update)
│   ├── feature_eng/
│   │   └── pipeline.py           FeatureEngineeringPipeline (VT → corr → MI|RF-RFE → scaler)
│   └── data_loader.py            DuckDB queries (CSV + SQLite)
├── inference/                    Runtime inference adapters + explainability
│   ├── adapters/
│   │   ├── csv_adapter.py        CSV upload → feature DataFrame
│   │   └── pcap_adapter.py       PCAP upload → NFStream flow extraction
│   └── explainers/               SHAP (Stage 2) + AE per-feature reconstruction error (Stage 3)
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

## Code navigation

| If you want to understand… | Start here |
|---|---|
| The 3-stage pipeline end-to-end | `docs/architecture.md` |
| Model protocols (what every classifier/detector must implement) | `core/base.py` — `BaseClassifier`, `BaseAnomalyDetector` |
| How XGBoost / MLP-AE are implemented | `core/classifiers/xgboost_clf.py`, `core/detectors/mlp_ae.py` |
| RF-RFE feature selection | `training/feature_eng/pipeline.py` |
| Initial training pipeline (CIC-IDS2017 → artefacts) | `training/pipelines/initial.py` |
| AE-only periodic retrain pipeline | `training/pipelines/fine_tune.py` |
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
