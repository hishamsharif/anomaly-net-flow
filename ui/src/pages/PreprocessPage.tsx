import { useEffect, useState } from 'react'

interface Stats {
  train_rows_raw: number | null
  train_rows_after_smote: number | null
  smote_before_total: number | null
  smote_after_total: number | null
  smote_n_minority: number | null
  smote_target_per_class: number | null
  class_distribution_train: Record<string, number>
  clf_scaler_type: string | null
  det_scaler_type: string | null
}

export default function PreprocessPanel() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [modelId, setModelId] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/pipeline/stats')
      .then(r => r.json())
      .then(d => {
        setModelId(d.model_id)
        setStats(d.stats)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return <div><h2>Preprocessing</h2><p>Loading…</p></div>

  if (modelId == null) {
    return (
      <div>
        <h2>Preprocessing</h2>
        <div className="card">
          <p style={{ color: 'var(--fg-muted)', margin: 0 }}>
            No trained model yet — preprocessing statistics are computed during initial training.
            Go to the <strong>Training</strong> tab to start.
          </p>
        </div>
      </div>
    )
  }

  const dist = stats?.class_distribution_train ?? {}
  const sortedCls = Object.entries(dist).sort((a, b) => b[1] - a[1])
  const maxCount = sortedCls[0]?.[1] ?? 1
  const nClasses = sortedCls.length
  const nMinority = stats?.smote_n_minority ?? 0
  const synthAdded = (stats?.smote_after_total ?? 0) - (stats?.smote_before_total ?? 0)

  return (
    <div>
      <h2>Preprocessing</h2>
      <p>
        Steps applied to the raw CIC-IDS2017 CSVs before training:
        numeric coercion, infinity replacement, and SMOTE resampling to correct class imbalance.
      </p>

      {/* ── Step 1: Numeric cleanup ── */}
      <h3>Step 1 — Numeric cleanup</h3>
      <div className="card">
        <p style={{ marginTop: 0 }}>
          All 78 CIC-IDS2017 columns are coerced to float32.
          Cells that cannot be parsed (e.g. strings in numeric columns) become <code>NaN</code>;
          infinity values (<code>±Inf</code>) and NaNs are replaced with <code>0</code>.
          This affects mainly <em>Flow Bytes/s</em> and <em>Flow Packets/s</em>, which CICFlowMeter
          occasionally records as <code>Infinity</code> for sub-millisecond flows.
        </p>
        <div className="metrics">
          <div className="metric">
            <div className="metric-label">Raw train rows</div>
            <div className="metric-value">
              {stats?.train_rows_raw != null ? stats.train_rows_raw.toLocaleString() : '—'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">Classes</div>
            <div className="metric-value">{nClasses}</div>
          </div>
          <div className="metric">
            <div className="metric-label">Replace strategy</div>
            <div className="metric-value" style={{ fontSize: 14 }}>NaN / ±Inf → 0</div>
          </div>
        </div>
      </div>

      {/* ── Step 2: Class imbalance ── */}
      <h3>Step 2 — Class imbalance (before SMOTE)</h3>
      <div className="card">
        <p style={{ marginTop: 0 }}>
          BENIGN dominates the dataset. Without correction, the classifier ignores rare attack
          classes. The bar below shows the raw imbalance before resampling.
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          {sortedCls.map(([cls, count]) => (
            <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 200, fontSize: 12, flexShrink: 0, textAlign: 'right',
                            color: cls === 'BENIGN' ? 'var(--fg-muted)' : 'var(--fg)' }}>
                {cls}
              </div>
              <div style={{ flex: 1, height: 14, background: 'var(--bg-card-inner, var(--border))',
                            borderRadius: 3, overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${(count / maxCount) * 100}%`,
                  background: cls === 'BENIGN'
                    ? 'var(--color-green, #22c55e)'
                    : 'var(--color-blue, #3b82f6)',
                  borderRadius: 3,
                  minWidth: 2,
                }} />
              </div>
              <div style={{ width: 100, fontSize: 12, color: 'var(--fg-muted)', flexShrink: 0 }}>
                {count.toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Step 3: SMOTE ── */}
      <h3>Step 3 — SMOTE resampling</h3>
      <div className="card">
        <p style={{ marginTop: 0 }}>
          SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples for
          minority classes up to a per-class cap. Applied only to the training split — validation
          and test data are never resampled (no leakage).
        </p>
        <div className="metrics" style={{ marginBottom: 12 }}>
          <div className="metric">
            <div className="metric-label">Before SMOTE</div>
            <div className="metric-value">
              {stats?.smote_before_total != null ? stats.smote_before_total.toLocaleString() : '—'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">After SMOTE</div>
            <div className="metric-value">
              {stats?.smote_after_total != null ? stats.smote_after_total.toLocaleString() : '—'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">Synthetic rows added</div>
            <div className="metric-value">{synthAdded > 0 ? synthAdded.toLocaleString() : '—'}</div>
          </div>
          <div className="metric">
            <div className="metric-label">Classes upsampled</div>
            <div className="metric-value">{nMinority > 0 ? nMinority : '—'}</div>
          </div>
          <div className="metric">
            <div className="metric-label">Cap per class</div>
            <div className="metric-value">
              {stats?.smote_target_per_class != null
                ? stats.smote_target_per_class.toLocaleString() : '—'}
            </div>
          </div>
        </div>
        <p style={{ fontSize: 13, color: 'var(--fg-muted)', margin: 0 }}>
          Cap prevents BENIGN (majority class) from generating millions of synthetic samples —
          the target is <code>min(majority_count, smote_max_per_class)</code>.
        </p>
      </div>

      {/* ── Step 4: Scaling ── */}
      <h3>Step 4 — Scaling (per model path)</h3>
      <div className="card">
        <p style={{ marginTop: 0 }}>
          XGBoost is tree-based and scale-invariant — no scaler applied.
          The MLP Autoencoder uses a QuantileTransformer to normalise sparse,
          heavy-tailed CIC-IDS2017 features into a Gaussian distribution.
          Scalers are fitted on training data only and applied identically to val/test.
        </p>
        <div className="metrics">
          <div className="metric">
            <div className="metric-label">Classifier scaler</div>
            <div className="metric-value" style={{ fontSize: 14 }}>
              {stats?.clf_scaler_type ?? '—'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">Detector scaler</div>
            <div className="metric-value" style={{ fontSize: 14 }}>
              {stats?.det_scaler_type ?? '—'}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
