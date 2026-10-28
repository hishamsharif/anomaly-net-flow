export default function AboutPage() {
  return (
    <div className="page">
      <h1>AnomalyNetFlow</h1>
      <p>
        Incremental network intrusion detection system with human-in-the-loop labelling.
        Trains on CIC-IDS2017, then improves continuously as analysts confirm or correct
        predictions from real traffic.
      </p>

      <h2>How it works</h2>
      <div className="card">
        <p><strong>1. Initial Training</strong> — Go to the <em>Train &amp; Evaluate</em> tab
        and click <em>Train on CIC-IDS2017</em>. The system trains a Classifier and
        an MLP Autoencoder on the bundled dataset (model types selectable).</p>
        <p><strong>2. Inference</strong> — Go to the <em>Inference</em> tab and upload a CSV
        (CIC-IDS2017 format) or PCAP file. Each flow gets a predicted label and an anomaly
        score through a 3-stage pipeline: rule engine → XGBoost → Autoencoder.</p>
        <p><strong>3. Analyst Feedback</strong> — Confirm correct predictions or override
        wrong ones. Each verdict is stored and counted toward the fine-tune threshold.</p>
        <p><strong>4. Incremental Fine-tune</strong> — Once 50 analyst verdicts accumulate,
        the <em>Fine-tune</em> button unlocks. Fine-tuning retrains on the full label store,
        incorporating analyst knowledge into the model.</p>
      </div>

      <h2>3-Stage Inference Pipeline</h2>
      <div className="card">
        <p>
          <strong>Stage 1 — Rule Engine:</strong> Fast deterministic rules (port ranges, known-good
          IP blocks, normal volume thresholds). Exits early for clearly benign traffic — reduces load
          on the ML stages.
        </p>
        <p>
          <strong>Stage 2 — Known-Attack Classifier:</strong> Trained with labelled data (15 CIC-IDS2017
          attack categories + BENIGN). Returns a named attack category and confidence score.
          Catches <em>known</em> threats it has seen during training.
        </p>
        <p>
          <strong>Stage 3 — Anomaly Detector:</strong> Trained on normal (benign) traffic only —
          no attack labels needed. Learns what "normal" looks like, then flags anything that deviates.
          Returns a per-flow anomaly score; flows above the p95 threshold are escalated.
          Catches <em>novel/zero-day</em> attacks the classifier has never seen.
          Per-feature reconstruction error (x − x̂)² provides built-in explainability.
        </p>
      </div>

      <h2>Why two detection stages?</h2>
      <div className="card">
        <p>
          The two ML stages catch different threat classes:
        </p>
        <ul>
          <li>
            <strong>Known-Attack Classifier (Stage 2)</strong> — high precision on known attacks.
            Needs labelled training data. Cannot detect attack variants it has not seen.
          </li>
          <li>
            <strong>Anomaly Detector (Stage 3)</strong> — catches zero-days and novel patterns
            with no labels required. Higher false-alarm rate (p95 = 5% floor by design).
          </li>
        </ul>
        <p>
          Together they cover both known and unknown threats — neither alone is sufficient.
        </p>
      </div>

      <h2>Models</h2>
      <div className="card">
        <p>
          <strong>Known-Attack Classifier (Stage 2) options:</strong>
        </p>
        <ul>
          <li><strong>XGBoost</strong> — gradient-boosted trees; fast, native SHAP explanations.
            Fine-tune = full cumulative retrain (trees are additive, not weight-based).</li>
          <li><strong>Random Forest</strong> — bootstrap aggregation; lower variance on small
            incremental datasets; balanced class weights handle imbalance without resampling.</li>
        </ul>
        <p>
          <strong>Anomaly Detector (Stage 3) options:</strong>
        </p>
        <ul>
          <li><strong>MLP Autoencoder</strong> — encoder-decoder trained to reconstruct normal flows;
            high reconstruction error = anomaly. Fine-tune = load weights → freeze encoder → train
            on new data + 10% replay buffer (prevents forgetting).</li>
          <li><strong>Isolation Forest</strong> — isolates anomalies by random partitioning; no
            training on gradients; fast. Good fallback when GPU is unavailable.</li>
        </ul>
        <p>
          <strong>Upgrade path:</strong> FT-Transformer (supervised) + Transformer-AE (unsupervised).
          Add one line to <code>core/registry.py</code> to expose a new model in this UI.
        </p>
      </div>

      <h2>Dataset</h2>
      <div className="card">
        <p>
          <strong>CIC-IDS2017</strong> — Canadian Institute for Cybersecurity, 2017.
          2.8M flows, 72 CICFlowMeter features, 15 attack classes + BENIGN.
          Day-based split: Mon–Thu train / Fri validation / Fri-afternoon test (time-ordered,
          not random — tests generalisation to unseen attack variants).
        </p>
      </div>

      <h2>Architecture</h2>
      <div className="card">
        <p>
          FastAPI · SQLAlchemy async · SQLite · DuckDB · React 18 + TypeScript + Vite.
          See <code>docs/architecture.md</code> in the project root for the full design document.
        </p>
      </div>
    </div>
  )
}
