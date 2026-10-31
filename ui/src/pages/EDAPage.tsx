import { useEffect, useState } from 'react'

interface Stats {
  class_distribution_train: Record<string, number>
  train_rows_raw: number | null
  val_rows: number | null
  test_rows: number | null
  classes: string[]
  ae_threshold: number | null
  training_duration_s: number | null
}

export default function EDAPanel() {
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

  if (loading) return <div><h2>EDA</h2><p>Loading…</p></div>

  if (modelId == null) {
    return (
      <div>
        <h2>Exploratory Data Analysis</h2>
        <div className="card">
          <p style={{ color: 'var(--fg-muted)', margin: 0 }}>
            No trained model yet — EDA statistics are extracted during initial training.
            Go to the <strong>Training</strong> tab to start.
          </p>
        </div>
      </div>
    )
  }

  const dist = stats?.class_distribution_train ?? {}
  const sortedCls = Object.entries(dist).sort((a, b) => b[1] - a[1])
  const totalRows = sortedCls.reduce((s, [, c]) => s + c, 0)
  const maxCount = sortedCls[0]?.[1] ?? 1
  const benignCount = dist['BENIGN'] ?? 0
  const attackCount = totalRows - benignCount
  const imbalanceRatio = benignCount > 0 && attackCount > 0
    ? (benignCount / attackCount).toFixed(1)
    : '—'

  const trainTotal = (stats?.train_rows_raw ?? 0) + (stats?.val_rows ?? 0) + (stats?.test_rows ?? 0)

  return (
    <div>
      <h2>Exploratory Data Analysis</h2>
      <p>
        Key statistics and class distribution from the CIC-IDS2017 training split.
        Understanding the imbalance between BENIGN and attack traffic is crucial —
        it drives the need for SMOTE resampling in the preprocessing step.
      </p>

      {/* ── Key observations ── */}
      <h3>Key observations</h3>
      <div className="card">
        <div className="metrics">
          <div className="metric">
            <div className="metric-label">Total flows</div>
            <div className="metric-value">{totalRows.toLocaleString()}</div>
          </div>
          <div className="metric">
            <div className="metric-label">BENIGN flows</div>
            <div className="metric-value">{benignCount.toLocaleString()}</div>
          </div>
          <div className="metric">
            <div className="metric-label">Attack flows</div>
            <div className="metric-value">{attackCount.toLocaleString()}</div>
          </div>
          <div className="metric">
            <div className="metric-label">Imbalance ratio</div>
            <div className="metric-value">{imbalanceRatio}:1</div>
          </div>
          <div className="metric">
            <div className="metric-label">Attack classes</div>
            <div className="metric-value">{sortedCls.length - 1}</div>
          </div>
          <div className="metric">
            <div className="metric-label">% BENIGN</div>
            <div className="metric-value">
              {totalRows > 0 ? ((benignCount / totalRows) * 100).toFixed(1) : '—'}%
            </div>
          </div>
        </div>
      </div>

      {/* ── Dataset split breakdown ── */}
      <h3>Split breakdown</h3>
      <div className="card">
        <p style={{ marginTop: 0, fontSize: 13, color: 'var(--fg-muted)' }}>
          Rows by dataset split — strict day boundary (no random shuffle).
        </p>
        {[
          { label: 'Train (Mon–Thu)', value: stats?.train_rows_raw, color: 'var(--color-green, #22c55e)' },
          { label: 'Val (Fri morning)', value: stats?.val_rows, color: 'var(--color-blue, #3b82f6)' },
          { label: 'Test (Fri afternoon)', value: stats?.test_rows, color: 'var(--color-orange, #f97316)' },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
            <div style={{ width: 160, fontSize: 13, flexShrink: 0 }}>{label}</div>
            <div style={{ flex: 1, height: 18, background: 'var(--bg-card-inner, var(--border))',
                          borderRadius: 4, overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: trainTotal > 0 && value != null ? `${(value / trainTotal) * 100}%` : '0%',
                background: color,
                borderRadius: 4,
                minWidth: value ? 2 : 0,
              }} />
            </div>
            <div style={{ width: 100, fontSize: 13, color: 'var(--fg-muted)', flexShrink: 0 }}>
              {value != null ? value.toLocaleString() : '—'}
            </div>
          </div>
        ))}
      </div>

      {/* ── Class distribution ── */}
      <h3>Class distribution — training split (before SMOTE)</h3>
      <div className="card">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          {sortedCls.map(([cls, count]) => (
            <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 220, fontSize: 12, flexShrink: 0, textAlign: 'right',
                            color: cls === 'BENIGN' ? 'var(--fg-muted)' : 'var(--fg)',
                            fontWeight: cls === 'BENIGN' ? 400 : 500 }}>
                {cls}
              </div>
              <div style={{ flex: 1, height: 16, background: 'var(--bg-card-inner, var(--border))',
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
              <div style={{ width: 110, fontSize: 12, color: 'var(--fg-muted)', flexShrink: 0 }}>
                {count.toLocaleString()} ({((count / totalRows) * 100).toFixed(2)}%)
              </div>
            </div>
          ))}
        </div>
        <p style={{ marginBottom: 0, marginTop: 12, fontSize: 13, color: 'var(--fg-muted)' }}>
          Heartbleed and Infiltration have &lt;50 samples total — they are excluded from
          cross-validation tuning due to insufficient class support.
        </p>
      </div>

      {/* ── AE threshold ── */}
      <h3>Anomaly detector baseline</h3>
      <div className="card">
        <p style={{ marginTop: 0 }}>
          The MLP Autoencoder is trained on BENIGN flows only and its anomaly decision threshold
          is calibrated on the validation split (Fri-morning benign flows). Threshold = p95
          reconstruction error, meaning 5% of normal flows are expected to be flagged as anomalies
          (the false-alarm floor).
        </p>
        <div className="metrics">
          <div className="metric">
            <div className="metric-label">AE threshold (p95)</div>
            <div className="metric-value">
              {stats?.ae_threshold != null ? stats.ae_threshold.toFixed(5) : '—'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">Training time</div>
            <div className="metric-value">
              {stats?.training_duration_s != null
                ? `${Math.round(stats.training_duration_s)}s` : '—'}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
