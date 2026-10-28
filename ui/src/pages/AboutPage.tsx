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
        and click <em>Train on CIC-IDS2017</em>. The system trains an XGBoost classifier and
        a Vanilla Autoencoder on the bundled dataset.</p>
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
          <strong>Stage 1 — Rule Engine:</strong> Fast deterministic rules flag obviously benign
          traffic (fast-path exit). Only suspicious flows proceed.
        </p>
        <p>
          <strong>Stage 2 — XGBoost Classifier:</strong> Supervised model trained on 15 CIC-IDS2017
          attack classes. Returns predicted label and confidence score.
        </p>
        <p>
          <strong>Stage 3 — Vanilla Autoencoder:</strong> Anomaly detector trained on benign flows.
          Returns a reconstruction error score; flows above the p95 threshold are flagged as anomalies.
        </p>
      </div>

      <h2>Models</h2>
      <div className="card">
        <p>
          <strong>Classifier (v1):</strong> XGBoost — fast, interpretable, native SHAP support.
          Fine-tune = full cumulative retrain (trees are additive, not weight-based).
        </p>
        <p>
          <strong>Anomaly Detector (v1):</strong> Vanilla Autoencoder (MLP, MSE loss) — simple
          and debuggable. Fine-tune = load previous weights → freeze encoder → train on new data
          + 10% replay buffer.
        </p>
        <p>
          <strong>Upgrade path:</strong> FT-Transformer (true weight-based fine-tuning),
          Transformer-AE (self-attention over feature tokens — correct inductive bias for tabular data).
          Add one line to <code>app/ml/registry.py</code> to expose a new model in this UI.
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
          See <code>doc/architecture.md</code> in the project root for the full design document.
        </p>
      </div>
    </div>
  )
}
