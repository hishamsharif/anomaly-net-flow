# AnomalyNetFlow — Solution Architecture

> Living document. Source of truth: Claude Docs https://claude.ai/artifact/H1ZQBwatmiRrM8a6ABNwKk

---

## 0. Problem Statement

### The SOC Automation Problem

Security Operations Centre (SOC) analysts face an overwhelming volume of network alerts — typically thousands per shift. The manual triage cycle is costly in three ways:

| Pain point | Impact |
|---|---|
| **Volume** — most alerts are repetitive known-attack patterns | Analysts spend the majority of their time on routine classification that could be automated |
| **Expert dependency** — novel/zero-day attacks require senior judgment | Knowledge lives in individuals, not in the system; when analysts rotate, that expertise is lost |
| **No feedback loop** — current tools classify but do not learn from analyst decisions | Analysts correct wrong predictions, but those corrections are discarded; the model never improves |

### What AnomalyNetFlow Does Differently

AnomalyNetFlow is designed around one core idea: **the model should learn from analyst experience, not just from labelled datasets**.

```
Day 1 — Initial Training
  Dataset (CIC-IDS2017) → Train XGBoost + AE → Deploy

Day N — Analyst reviews alerts
  Alert stream → Pipeline prediction → Analyst confirms or overrides → Verdict stored

Day N+50 — Fine-tune unlocks
  Accumulated analyst verdicts → Retrain on combined dataset + verdicts
  → Model now "knows" what this analyst team considers an attack

Repeat
```

Three stages automate different layers of the SOC triage workload:

| Stage | What it automates | What stays with the analyst |
|---|---|---|
| Rule Engine (Stage 1) | Obvious benign traffic — clears the fast path instantly | Nothing — fully automated |
| Known-Attack Classifier (Stage 2) | Repeat attack patterns seen in training data (XGBoost) | Verdicts on classifier-flagged high-confidence alerts are stored as learning signal |
| Anomaly Detector (Stage 3) | Flags traffic that deviates from the learnt benign baseline — these are *potential* novel/zero-day attacks | Analyst reviews, confirms or labels → this verdict teaches the model what was actually novel |

The **incremental fine-tune loop** is the key differentiator. Every analyst verdict — confirmation or correction — is stored in the label store. Once 50 verdicts accumulate (configurable), the system prompts: "Fine-tune available." The analyst triggers retraining on all historical data plus their verdicts. Over time:

- Novel attacks seen and labelled by analysts enter the known-attack classifier
- The model's knowledge base grows in proportion to analyst experience captured
- The routine-classification burden on analysts decreases sprint by sprint

### The Concept Drift Problem

Most NIDS research frames the problem as a classification accuracy question — *how good is the model on a held-out test set?* This framing misses the central challenge of real-world deployment:

> **A model trained today will degrade over time because attack patterns evolve. The core problem is not accuracy on a static benchmark — it is building a system that adapts to an evolving threat landscape without requiring a full manual retrain each time a new attack variant is observed.**

This is the **concept drift** problem in intrusion detection. Attack toolkits are updated, new CVEs are discovered, adversaries change their timing, packet sizes, and flag patterns to evade known signatures. A classifier trained on last year's CIC-IDS2017 data will encounter traffic next year that its feature distributions cannot represent.

AnomalyNetFlow addresses concept drift at three levels:

| Level | Mechanism | How it handles drift |
|---|---|---|
| **Known attacks** | XGBoost (supervised, CIC-IDS2017 Mon–Thu) | High accuracy on training-distribution attacks; degrades gracefully as variants drift |
| **Novel / drifted attacks** | MLP-AE (unsupervised, benign baseline) | Learns *normal* not *attacks* — flags anything that deviates, regardless of whether it matches a known signature |
| **Adaptive re-labelling** | Analyst verdicts → incremental fine-tune | Incorporates new ground truth as it is observed; analyst labels are the adaptation signal |

The staged pipeline means drift in the classifier does not compromise overall detection — the anomaly detector provides a safety net for anything that has drifted outside the classifier's training distribution. As analysts label those novel detections, the classifier learns the new patterns and takes them over in the next fine-tune cycle.

**Why the day-based split demonstrates this:** CIC-IDS2017 is captured Mon–Fri. Friday contains attack variants (different tools, slightly different traffic profiles) not seen in Mon–Thu training. The classifier's AMR (Attack Miss Rate) on Friday data is ≈ 99% — it misses almost everything new. This is not a weakness of XGBoost; it is a controlled demonstration of concept drift within the same week. Stage 3 catches what Stage 2 misses; the fine-tune loop closes the gap over time.

### Why CIC-IDS2017 as the Starting Dataset

CIC-IDS2017 (University of New Brunswick, 2017) provides 2.8 million labelled flows across 15 attack classes captured across five working days. It is used as the *initial* training corpus — the model's Day 0 knowledge before any analyst interaction. The dataset:

- Covers the most common attack types a SOC encounters (DoS, DDoS, PortScan, Brute Force, Web Attacks, Botnet, Infiltration)
- Is structured as a day-based chronological capture — Mon–Thu trains the classifier; Friday tests generalisation to unseen attack variants, mirroring the real-world condition where a newly-deployed NIDS encounters attack patterns it has never been trained on
- Is in CIC-IDS2017 CSV format (78 statistical flow features from CICFlowMeter), making it compatible with the PCAP-to-CSV inference pipeline

The day-based split is critical to honest evaluation: it surfaces the known-attack classifier's generalisation gap on Friday data (AMR ≈ 99%), which is exactly the gap Stage 3 is designed to fill.

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

## 1b. Explicit Assumptions and Parameter Thresholds

### Assumptions

#### Dataset assumptions
| Assumption | Rationale | Risk if wrong |
|---|---|---|
| CIC-IDS2017 labels are ground truth | Provided by UNB researchers who injected attacks into a controlled environment | Label noise would silently corrupt F1 scores — investigate if metrics look implausibly high |
| 78 statistical flow features are sufficient for classification | CICFlowMeter features capture directionality, timing, flag counts, IAT distributions — the key discriminants between attack types | Encrypted C2 traffic / evasive attacks that mimic benign timing may not be separable; packet-byte DL needed for those |
| Day-based split (Mon–Thu train / Fri test) simulates real deployment | Friday captures include attack variants from the same campaign but with different tooling — a valid proxy for concept drift within a week | Over-optimistic on long-horizon drift (months); multi-week captures needed for drift stress-test |
| BENIGN class is representative of the deployment environment | Training benign flows are typical enterprise HTTP/HTTPS/DNS traffic | A network with unusual baseline traffic (e.g. industrial OT) will produce more false positives from the AE |
| Analyst verdicts are reliable | System design assumes analysts are correct when they confirm or override predictions | Adversarial or negligent analysts could poison the label store; no verdict-validation layer in v1 |
| Attack classes with < cv×10 samples are excluded from CV | Heartbleed (~11 rows), Web Attack SQL Injection (~1 row) cannot survive a 3-fold CV subsample | These classes are still present in initial training — they are excluded only from the hyperparameter search, not the deployed model |

#### Model assumptions
| Assumption | Rationale |
|---|---|
| Reconstruction error (MSE) is a valid anomaly score for tabular flow data | The AE learns the manifold of benign traffic; flows that deviate from it produce higher reconstruction error regardless of the specific attack mechanism |
| p95 benign validation threshold → FAR ≤ 5% at deployment | Holds only if deployment traffic distribution matches training benign distribution; FAR rises if deployment is noisier |
| XGBoost full cumulative retrain is feasible at fine-tune frequency | At 50 verdicts/cycle on a 2.8M-row dataset the retrain takes ~60 s; still viable up to ~10M rows before warm-start trees are needed |
| SMOTE on train-only split does not leak into validation | SMOTE is applied after the train/val/test split; val and test rows are never oversampled |

---

### Parameter Thresholds

#### Stage 1 — Rule Engine (`api/routers/predict.py :: _rule_verdict`)

| Rule | Condition | Verdict | Rationale |
|---|---|---|---|
| DNS / DHCP fast-path | `Destination Port ∈ {53, 67, 68}` AND `Total Fwd Packets ≤ 1` AND `Flow Duration < 100 µs` | BENIGN (skip AI) | Single-packet sub-100µs DNS/DHCP queries are structurally impossible to be attacks — clears the most common traffic type instantly |
| All other traffic | — | PASS (proceed to Stage 2 + 3) | |

*The rule engine is intentionally minimal (3 conditions). It is a fast-path for obvious benign, not a replacement for the classifier.*

#### Stage 2 — Known-Attack Classifier

**Consensus threshold** (`api/routers/predict.py :: _consensus`)

| Condition | Output label |
|---|---|
| `clf_label ≠ BENIGN` AND `clf_confidence ≥ 0.60` | `KNOWN_ATTACK` |
| `is_anomaly = True` (Stage 3) | `NOVEL_ATTACK` |
| Neither | `SAFE` |

> **Why 0.60?** Below 0.6 the classifier is uncertain — typically a flow at the boundary of two attack classes. Forcing a KNOWN_ATTACK label at low confidence adds noise to analyst alerts. Better to let Stage 3 flag it as NOVEL_ATTACK (higher-priority review queue) than mislabel it with false certainty.

**XGBoost defaults** (`core/classifiers/xgboost_clf.py`) — set by `RandomizedSearchCV` (n_iter=20, cv=3, stratified 100K subsample, Best CV F1 macro = 0.8423):

| Parameter | Value | Why chosen |
|---|---|---|
| `n_estimators` | 489 | Search converged toward higher values; 12-class problem benefits from more boosting rounds |
| `max_depth` | 5 | Shallower than default 6; reduces overfitting on imbalanced raw class distribution |
| `learning_rate` | 0.136 | Moderate step size — balances convergence speed and generalisation |
| `subsample` | 0.683 | Stochastic row subsampling per tree adds regularisation; speeds training on 2M-row dataset |
| `colsample_bytree` | 0.964 | Near-full feature sampling — most features are informative after VT + correlation pruning |
| `min_child_weight` | 6 | Prevents splits on very rare attack classes with few natural representatives |
| `gamma` | 0.129 | Small minimum loss reduction; prunes useless splits without cutting informative ones |
| `tree_method` | hist | CPU-optimised histogram algorithm; change `device="cuda"` for GPU |

**Random Forest defaults** (`core/classifiers/random_forest_clf.py`):

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 300 | Sufficient ensemble size for tabular data without excessive training time |
| `max_depth` | None (unlimited) | Bootstrap aggregation provides implicit regularisation; depth restriction adds little |
| `min_samples_leaf` | 2 | Prevents single-sample leaves; reduces overfitting on rare attack rows |
| `class_weight` | `balanced` | Handles CIC-IDS2017 imbalance without SMOTE overhead (used as an alternative to XGBoost) |

#### Stage 3 — Anomaly Detector (MLP Autoencoder)

**Architecture** (`core/detectors/mlp_ae.py`):

| Parameter | Value | Rationale |
|---|---|---|
| `hidden_dims` | `[128, 64, 32]` | Bottleneck encoder forces a compressed representation; decoder mirrors the encoder |
| `epochs` | 30 | Sufficient for convergence on benign-only training; early stopping not yet implemented |
| `batch_size` | 512 | Balanced between gradient noise and GPU/CPU memory |
| `learning_rate` | 1e-3 | Adam default; well-suited to MSE loss on normalised tabular data |
| `anomaly_threshold_percentile` | **95.0** (p95) | 5% FAR floor by construction on benign validation data (see §0 concept drift rationale) |
| `freeze_epochs` (fine-tune) | 5 | Encoder frozen for first 5 epochs of fine-tune; prevents catastrophic forgetting of the benign manifold |

**Scaler choice for AE:** QuantileTransformer (maps any distribution to [0,1]). RobustScaler rejected because CIC-IDS2017 sparse features (e.g. `Flow Bytes/s`) have IQR ≈ 0, making `scale_ = 1` — outliers reach the AE unscaled and produce thresholds of ~38M (unusable). QuantileTransformer produces thresholds of ~0.01–0.1 (calibrated FAR).

#### Feature Engineering (`training/feature_eng/pipeline.py` + `api/config.py`)

| Stage | Parameter | Default | Env override | Effect |
|---|---|---|---|---|
| VarianceThreshold | `variance_threshold` | **0.0** | `VARIANCE_THRESHOLD` | Drops only perfectly constant features (0 = weakest filter; raise to 0.01–0.05 for stricter pruning) |
| Correlation pruning | `correlation_threshold` | **0.95** | `CORRELATION_THRESHOLD` | Drops one from each pair with `\|Pearson r\| > 0.95`; prevents redundant features from amplifying importance scores |
| Feature selection (optional) | `feature_selection_k` | **0** (disabled) | `FEATURE_SELECTION_K` | 0 = keep all features that pass steps 1–2; set to e.g. 30 to select top-30 |
| Feature selection method | `feature_selection_method` | **mi** | `FEATURE_SELECTION_METHOD` | `mi` — SelectKBest(mutual\_info\_classif), fast, marginal; `rfe` — RF-RFE, captures feature interactions (literature: 80→20 features, no F1 loss on CIC-IDS2017); `none` — skip step 3 entirely |
| Classifier scaler | `scaler_type` | **robust** | `SCALER_TYPE` | RobustScaler for classifier path (XGBoost is scale-invariant, but RF benefits from normalised inputs when class_weight is used) |
| Detector scaler | hardcoded | **quantile** | — | QuantileTransformer — see AE scaler choice above; not configurable to prevent FAR breakage |

#### Human-in-the-Loop Parameters (`api/config.py`)

| Parameter | Default | Env override | Effect |
|---|---|---|---|
| `fine_tune_k_threshold` | **50** | `FINE_TUNE_K_THRESHOLD` | Minimum analyst verdicts since last training before Fine-tune is unlocked in the UI |
| `ae_replay_buffer_fraction` | **0.10** | `AE_REPLAY_BUFFER_FRACTION` | Fraction of previous training data replayed during AE fine-tune to prevent catastrophic forgetting |
| `ae_freeze_epochs` | **5** | `AE_FREEZE_EPOCHS` | Epochs to freeze AE encoder during fine-tune before full unfreeze |
| `smote_max_per_class` | **50,000** | `SMOTE_MAX_PER_CLASS` | Cap on synthetic samples per minority class; uncapped SMOTE on CIC-IDS2017 produces ~17M rows (8+ h training) |
| `max_upload_mb` | **100** | `MAX_UPLOAD_MB` | Max inference upload size for CSV/PCAP files |

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
| Anomaly detector | **MLP Autoencoder** (fully-connected, MSE loss) · **Isolation Forest** (sklearn, no GPU) · **TransferAutoencoder** (UNSW-NB15 pretrain → CIC-IDS2017 fine-tune; see §5) | **Transformer-AE / VAE** — self-attention over feature tokens |
| Feature selection | **MI** — SelectKBest(mutual\_info\_classif) · **RF-RFE** — RandomForest-driven recursive elimination (see §5) | — |
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
Flow features (78 raw cols → 47 after feature engineering)
        ↓
┌───────────────────┐
│  Rule Engine      │  DNS/DHCP fast-path rule  →  BENIGN (skip AI)
│  (Stage 1)        │  dst ∈ {53,67,68} AND pkts≤1 AND dur<100µs
└────────┬──────────┘
         │ PASS (all other traffic)
┌────────▼──────────┐
│  Known-Attack     │  Supervised classifier   →  predicted label + confidence
│  Classifier       │  15 CIC-IDS2017 classes  (XGBoost · RF)
│  (Stage 2)        │  threshold: clf_conf ≥ 0.60 → KNOWN_ATTACK
└────────┬──────────┘
         ↓  (always — both stages run in parallel)
┌────────▼──────────┐
│  Anomaly Detector │  Reconstruction error (MSE)  →  above p95 threshold
│  (Stage 3)        │  MLP-AE · IsolationForest · TransferAE
└────────┬──────────┘
         ↓
  Consensus → SAFE / KNOWN_ATTACK / NOVEL_ATTACK
  (see §1b Stage 2 consensus threshold for full decision table)
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

### RF-RFE Feature Selection

Standard mutual information (MI) ranks features **independently** — it scores each feature's marginal correlation with the label but ignores how features interact. Random Forest-driven Recursive Feature Elimination (RF-RFE) uses the RF's impurity-based importance scores, which account for **feature interactions** through the tree-split mechanism.

**How RF-RFE works in `training/feature_eng/pipeline.py`:**

```
After VT + correlation pruning (N features remain):
  1. Fit RandomForest(n_estimators=100, max_depth=8, balanced)
     on the training split
  2. RFE iteratively removes step=⌊(N−k)/10⌋ features per round
     (adaptive step ≈ 10 elimination rounds regardless of N)
  3. After k features remain, apply the support mask to train/val/test
```

| Method | Strength | When to use |
|---|---|---|
| `mi` (default) | Fast (~seconds) | First pass; good baseline |
| `rfe` | Captures interactions; literature reports 80→20 features with no F1 loss on CIC-IDS2017 (Sharafaldin et al.) | When MI selection hurts known-attack F1 or when interpretability matters |
| `none` | Keeps all post-correlation features | Ablation; verifying engineering value |

**Configuration:** set `FEATURE_SELECTION_METHOD=rfe` (+ `FEATURE_SELECTION_K=20`) in `.env` before triggering training. The selected method is recorded in `metadata.json` → `selection_method` and surfaced in the Training tab Features sub-tab.

---

### TransferAutoencoder — Cross-Dataset Transfer Learning

**Motivation:** MLP-AE trained from scratch on CIC-IDS2017 learns the benign manifold from one capture environment. Transfer learning from UNSW-NB15 gives the encoder a richer prior — it has already seen a broader range of benign flow patterns — and improves calibration, especially at the tail of the benign reconstruction-error distribution where the anomaly threshold lives.

**Architecture (`core/detectors/transfer_ae.py`):**

```
Phase 1 — Pretrain on UNSW-NB15 (source domain, src_dim features):
  X_src → _SourceAENet(src_dim → [128,64,32] → src_dim)
  Loss: MSE(X_src, X̂_src)

Phase 2 — Transfer to CIC-IDS2017 (target domain, tgt_dim features):
  X_tgt → FeatureAlignment(tgt_dim → src_dim)   ← always trainable
          → encoder[128,64,32]                  ← frozen for freeze_epochs
          → decoder_tgt([32,64,128] → tgt_dim)  ← always trainable
  Loss: MSE(X_tgt, X̂_tgt)
```

The **FeatureAlignment layer** (linear projection, no activation) maps the target feature space into the source encoder's input space. This allows the pretrained encoder to process CIC-IDS2017 flows without modifying its weights during the frozen phase.

**4-Experiment comparison (A2 ablation):**

| Experiment | Detector | Config | What it tests |
|---|---|---|---|
| E1 | `isolation_forest` | Default | Baseline — classical anomaly detection, no deep learning |
| E2 | `mlp_ae` | Default | Deep AE trained on CIC-IDS2017 only — no transfer |
| E3 | `transfer_ae` + `pretrain()` | `freeze_epochs=30` (encoder never unfreezes) | Pretrained frozen — does UNSW-NB15 pretraining alone improve threshold? |
| E4 | `transfer_ae` + `pretrain()` | `freeze_epochs=5` (default) | Full fine-tune — best of both: pretrained prior + target domain adaptation |

**Usage (trigger E4 from ops.sh or UI):**

```bash
# UI: Training tab → Detector = TransferAutoencoder → Train
# ops.sh:
bash ops.sh train initial xgboost transfer_ae
```

*Note: E3/E4 require UNSW-NB15 benign flows to be passed to `pretrain()` before `fit()`. A `/api/training/pretrain` endpoint or a `PRETRAIN_SOURCE_DIR` config key is the planned wiring path (M2 roadmap). Without calling `pretrain()`, `TransferAutoencoder.fit()` falls back silently to the same behaviour as E2.*

**Serialisation:** `save()`/`load()` persist both the source AE state dict and the transfer AE state dict in a single `detector.joblib` file. The `_pretrained` flag allows `load()` to reconstruct the correct network topology.

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
| M1 Core | XGBoost + MLP-AE + RF-RFE + TransferAutoencoder, label store, analyst feedback, fine-tune loop |
| M2 Pretrain wiring | `PRETRAIN_SOURCE_DIR` config key + `/api/training/pretrain` endpoint to load UNSW-NB15 benign flows and call `TransferAutoencoder.pretrain()` before initial training |
| M3 Explainability | SHAP waterfall charts in alert detail panel |
| M4 Multi-dataset | UNSW-NB15 + CICIDS2018 as additional source datasets; dataset-specific feature registries |
| M5 Auth | JWT + analyst vs admin roles |
| M6 Export | PDF/CSV report generation |
| M7 Streaming | WebSocket inference; sub-100 ms per-flow latency |
| M8 Packet-based DL | 1D-CNN / LSTM on raw packet bytes — flags active connections before the flow completes (no 15 s wait for statistical feature aggregation); enables sub-second detection of zero-days and encrypted threats. |

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
│   └── detectors/                 # mlp_ae.py · isolation_forest.py · transfer_ae.py
├── training/                      # Batch training processes
│   ├── pipelines/
│   │   ├── initial.py             # CIC-IDS2017 initial training
│   │   └── fine_tune.py           # Incremental fine-tune from label store
│   ├── feature_eng/
│   │   └── pipeline.py            # FeatureEngineeringPipeline (VT → corr → MI|RF-RFE → scaler)
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
├── ops.sh                         # start | stop | restart | status | logs | ui | train [mode] [clf] [det] [--follow]
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
- **Feature selection method default = `mi`** (`FEATURE_SELECTION_METHOD`). MI (SelectKBest) is the default because it is fast and sufficient for the baseline. Switching to `rfe` (RF-RFE) is a drop-in replacement that captures feature interactions; literature reports 80→20 with no F1 loss. The selected method is saved to `metadata.json → selection_method` for reproducibility.
- **TransferAutoencoder** is registered as `transfer_ae` in `core/registry.py` and follows the identical `fit() / score() / is_anomaly() / save() / load()` interface as `MLPAutoencoder`. The pretrain step (UNSW-NB15 benign flows) is optional — without it, the class degrades silently to the same behaviour as plain MLP-AE. Source + target state dicts are both serialised in a single `detector.joblib` for portability.
- **TransferAE fallback without pretrain**: `TransferAutoencoder.fit()` without a prior `pretrain()` call builds a plain `_SourceAENet` on the target features (same as `MLPAutoencoder`). This makes E2 vs E3/E4 comparisons possible by choosing whether to call `pretrain()` before `fit()`, with no code change in the training pipeline.

---

## Expected behaviour — metrics on initial training

| Metric | Typical value | Why |
|---|---|---|
| `known_attack_f1` | ~0.90+ | Classifier on Mon-Thu attack classes only — valid performance measure |
| `f1` (macro, all classes) | ~0.07 | Test set = Friday flows; most Friday attack types not in training → F1=0 for those classes; macro average divided by 13 |
| `amr` | ~0.99 | 99% of Friday attacks misclassified as BENIGN by classifier — **Stage 3 AE catches these** |
| `far` | ~0.0002 | Only 0.02% of benign flows falsely flagged — classifier is precise |
| `ae_threshold` | ~0.003 | p95 of benign reconstruction errors after QuantileTransformer — a flow above this is anomalous |

**The low macro F1 is a feature, not a bug.** It demonstrates why Stage 3 is needed: the Known-Attack Classifier handles known attacks well; the Anomaly Detector catches what it misses (novel/zero-day).
