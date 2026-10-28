# AnomalyNetFlow

Incremental network intrusion detection with human-in-the-loop labelling.  
XGBoost + MLP Autoencoder · CIC-IDS2017 · Active learning via analyst feedback.

Architecture: [`doc/architecture.md`](doc/architecture.md)

---

## Prerequisites

Before running anything, you need **three things only**:

| # | What | Where to get it |
|---|---|---|
| 1 | **Miniconda** (or Anaconda) | https://docs.conda.io/en/latest/miniconda.html |
| 2 | **Node.js ≥ 20** | https://nodejs.org |
| 3 | **CIC-IDS2017 CSV files** placed in `data/CIC-IDS2017/` | https://www.unb.ca/cic/datasets/ids-2017.html |

No Redis. No Docker. No database server.  
SQLite is file-based and created automatically on first run.

---

## Quick start

### Step 1 — Run setup (once)

```bash
bash setup.sh
```

This will:
- Create a conda environment (`dl_env`, Python 3.11)
- Install all Python dependencies (`requirements.txt`)
- Install UI dependencies (`npm install`)
- Create `.env` from `.env.example`
- Check for the CIC-IDS2017 dataset and warn if missing

### Step 2 — Start the API

```bash
bash ops.sh start
```

API available at **http://localhost:8000** · Docs at **http://localhost:8000/docs**

### Step 3 — Start the UI *(separate terminal)*

```bash
bash ops.sh ui
```

UI available at **http://localhost:5173**

---

## First run: initial training

The **Train & Evaluate** tab shows a *"Train on CIC-IDS2017"* button because no model is active yet.

Click it — or from the command line:

```bash
bash ops.sh train initial
```

Training reads the CIC-IDS2017 CSVs directly (no import), trains your selected Classifier + Anomaly Detector, and streams live progress via SSE. The Training tab updates with metrics (F1, PR-AUC, Recall, FAR, AMR) when done.

---

## Operations reference

```
bash setup.sh              First-time setup (run once)

bash ops.sh start          Start the API
bash ops.sh stop           Stop the API
bash ops.sh restart        Restart the API
bash ops.sh status         Show running status + health check
bash ops.sh logs           Tail the API log  (Ctrl-C to exit)
bash ops.sh ui             Start the React UI  (port 5173)
bash ops.sh train          Trigger initial training
bash ops.sh train fine_tune  Trigger incremental fine-tune
bash ops.sh open           Open the UI in your browser
bash ops.sh help           Show all commands
```

---

## How the data works

```
data/CIC-IDS2017/*.csv          Original dataset — stays as CSV files
        │
        │   DuckDB reads directly at training time
        ▼
  Initial model trained (Classifier + Anomaly Detector)
        │
  Analyst uploads CSV or PCAP → inference results shown
        │
  Analyst confirms or corrects labels → written to SQLite
        │
  anomaly_net_flow.db             SQLite (analyst verdicts only)
        │
        │   DuckDB reads at fine-tune time
        ▼
  Incremental fine-tune (unlocks after 50 analyst verdicts)
```

---

## Human-in-the-loop flow

1. **Inference tab** — upload a CSV or PCAP file
2. The 3-stage pipeline (rule engine → Classifier → Anomaly Detector) predicts a label for each flow
3. The analyst reviews flagged flows and submits a verdict:
   - **Confirm** — prediction was correct
   - **Override** — select the correct label
4. Both verdict types are stored and count toward the fine-tune threshold (default: 50)
5. When the threshold is reached, the **Fine-tune** button unlocks on the Training tab

---

## API endpoints

Base URL: `http://localhost:8000` · Full docs: `http://localhost:8000/docs`

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/api/models/active` | Active model or null |
| GET | `/api/models/` | All model versions + metrics |
| POST | `/api/training/start` | Start training — returns `run_id` |
| GET | `/api/training/events/{run_id}` | SSE stream of live progress |
| GET | `/api/training/available-types` | Available classifier + detector types |
| POST | `/api/predict/batch` | Upload CSV or PCAP for batch inference |
| GET | `/api/alerts/` | Inference alerts (paginated) |
| POST | `/api/alerts/{id}/feedback` | Submit analyst verdict |
| GET | `/api/alerts/metrics` | Live F1 / FAR / AMR + verdict count |

---

## Project structure

```
anomaly-net-flow/
├── setup.sh                    First-time setup
├── ops.sh                      All runtime operations
├── app/                        FastAPI backend
│   ├── main.py
│   ├── config.py               Settings (.env / environment variables)
│   ├── database.py             SQLAlchemy async (SQLite) + DuckDB helper
│   ├── db/models.py            ORM models (labelled_flows, alerts, model_versions …)
│   ├── tasks.py                In-process background training (ThreadPoolExecutor + SSE)
│   ├── ml/
│   │   ├── registry.py         Add one line here to expose a new model in the UI
│   │   ├── classifiers/        XGBoostClassifier · RandomForestClassifier
│   │   └── detectors/          MLPAutoencoder · IsolationForestDetector
│   └── routers/                predict, training, models, alerts, features
├── training/
│   ├── pipelines/
│   │   ├── initial.py          CIC-IDS2017 CSVs → XGBoost + AE
│   │   └── fine_tune.py        Analyst verdicts from SQLite → incremental update
│   ├── adapters/
│   │   ├── csv_inference.py    CSV upload → feature DataFrame
│   │   └── pcap_adapter.py     PCAP upload → flow extraction
│   └── data_loader.py          DuckDB queries (CSV + SQLite)
├── ui/                         React 18 + TypeScript + Vite 5
│   └── src/pages/              AboutPage · TrainingPage · InferencePage
├── data/CIC-IDS2017/           Place dataset CSV files here (not in git)
├── models/                     Saved model artefacts  (created at runtime)
├── doc/architecture.md         Full system architecture
├── requirements.txt
└── .env.example
```

---

## Configuration

All settings have working defaults — no configuration needed for a local run.  
To override, copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./anomaly_net_flow.db` | SQLite database path |
| `BUNDLED_DATASET_DIR` | `data/CIC-IDS2017` | CIC-IDS2017 CSV folder |
| `FINE_TUNE_K_THRESHOLD` | `50` | Analyst verdicts needed to unlock fine-tune |
| `AE_REPLAY_BUFFER_FRACTION` | `0.1` | Fraction of old data replayed during AE fine-tune |

---

## Extending with a new model

1. Create `app/ml/classifiers/my_model.py` implementing `BaseClassifier` (`app/ml/base.py`)
2. Add one line to `app/ml/registry.py`:
   ```python
   CLASSIFIERS["my_model"] = MyModelClassifier
   ```
3. The model appears automatically in the Training tab and `GET /api/training/available-types`
