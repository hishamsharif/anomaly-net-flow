import { useEffect, useState } from 'react'

interface FileInfo {
  name: string
  size_mb: number
}
interface DatasetInfo {
  data_dir: string
  csv_count: number
  total_size_mb: number
  files: FileInfo[]
}
interface ClassDist {
  [cls: string]: number
}
interface PipelineStats {
  model_id: number | null
  stats: {
    class_distribution_train: ClassDist
    train_rows_raw: number | null
    val_rows: number | null
    test_rows: number | null
    classes: string[]
  } | null
}

export default function DatasetPanel() {
  const [dataset, setDataset] = useState<DatasetInfo | null>(null)
  const [pipeline, setPipeline] = useState<PipelineStats | null>(null)

  useEffect(() => {
    fetch('/api/pipeline/dataset').then(r => r.json()).then(setDataset).catch(() => {})
    fetch('/api/pipeline/stats').then(r => r.json()).then(setPipeline).catch(() => {})
  }, [])

  const classDist = pipeline?.stats?.class_distribution_train ?? {}
  const totalRows = Object.values(classDist).reduce((a, b) => a + b, 0)
  const sortedClasses = Object.entries(classDist).sort((a, b) => b[1] - a[1])
  const maxCount = sortedClasses[0]?.[1] ?? 1

  return (
    <div>
      <h2>Dataset</h2>
      <p>
        CIC-IDS2017 — Canadian Institute for Cybersecurity, 2017. Five days of labelled network
        traffic captured at CICIDS with CICFlowMeter. Monday is benign-only; Tuesday–Friday
        introduce 15 attack categories.
      </p>

      {/* ── File inventory ── */}
      <h3>CSV files in <code>{dataset?.data_dir ?? 'data/CIC-IDS2017'}</code></h3>
      {!dataset && <p style={{ color: 'var(--fg-muted)' }}>Loading…</p>}
      {dataset && dataset.csv_count === 0 && (
        <div className="card" style={{ borderColor: 'var(--danger)' }}>
          <strong style={{ color: 'var(--danger)' }}>No CSV files found.</strong>
          <p>Place CIC-IDS2017 CSV files in <code>{dataset.data_dir}</code> then run training.</p>
        </div>
      )}
      {dataset && dataset.csv_count > 0 && (
        <div className="card">
          <div className="metrics" style={{ marginBottom: 12 }}>
            <div className="metric">
              <div className="metric-label">CSV files</div>
              <div className="metric-value">{dataset.csv_count}</div>
            </div>
            <div className="metric">
              <div className="metric-label">Total size</div>
              <div className="metric-value">{dataset.total_size_mb.toLocaleString()} MB</div>
            </div>
            {pipeline?.stats?.train_rows_raw != null && (
              <div className="metric">
                <div className="metric-label">Train rows</div>
                <div className="metric-value">{pipeline.stats.train_rows_raw.toLocaleString()}</div>
              </div>
            )}
            {pipeline?.stats?.val_rows != null && (
              <div className="metric">
                <div className="metric-label">Val rows</div>
                <div className="metric-value">{pipeline.stats.val_rows.toLocaleString()}</div>
              </div>
            )}
            {pipeline?.stats?.test_rows != null && (
              <div className="metric">
                <div className="metric-label">Test rows</div>
                <div className="metric-value">{pipeline.stats.test_rows.toLocaleString()}</div>
              </div>
            )}
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>File</th>
                  <th style={{ textAlign: 'right' }}>Size (MB)</th>
                  <th>Day / split</th>
                </tr>
              </thead>
              <tbody>
                {dataset.files.map(f => (
                  <tr key={f.name}>
                    <td style={{ fontSize: 13 }}>{f.name}</td>
                    <td style={{ textAlign: 'right', fontSize: 13 }}>{f.size_mb}</td>
                    <td style={{ fontSize: 12, color: 'var(--fg-muted)' }}>
                      {splitLabel(f.name)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Day-based split strategy ── */}
      <h3>Train / Val / Test split</h3>
      <div className="card">
        <p style={{ marginTop: 0 }}>
          The dataset is split by capture day, not randomly. This tests generalisation to
          attacks recorded on different days — a stricter evaluation than a random 80/10/10 row split.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Split</th><th>Days / files</th><th>Purpose</th></tr>
            </thead>
            <tbody>
              <tr>
                <td><span className="badge badge-green">Train</span></td>
                <td style={{ fontSize: 13 }}>Monday – Thursday</td>
                <td style={{ fontSize: 13 }}>Supervised classifier + AE training</td>
              </tr>
              <tr>
                <td><span className="badge badge-yellow">Val</span></td>
                <td style={{ fontSize: 13 }}>Friday Morning (WorkingHours)</td>
                <td style={{ fontSize: 13 }}>AE anomaly threshold calibration (p95 benign)</td>
              </tr>
              <tr>
                <td><span className="badge badge-red">Test</span></td>
                <td style={{ fontSize: 13 }}>Friday Afternoon (DDoS + PortScan)</td>
                <td style={{ fontSize: 13 }}>Final evaluation — novel attacks unseen in training</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Class distribution ── */}
      <h3>Class distribution (training split)</h3>
      {pipeline?.model_id == null && (
        <div className="card">
          <p style={{ color: 'var(--fg-muted)', margin: 0 }}>
            Run initial training first — class distribution is extracted during training.
          </p>
        </div>
      )}
      {sortedClasses.length > 0 && (
        <div className="card">
          <p style={{ marginTop: 0, fontSize: 13, color: 'var(--fg-muted)' }}>
            {totalRows.toLocaleString()} total rows across {sortedClasses.length} classes
            (before SMOTE resampling)
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {sortedClasses.map(([cls, count]) => (
              <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ width: 220, fontSize: 12, flexShrink: 0, textAlign: 'right',
                              color: cls === 'BENIGN' ? 'var(--fg-muted)' : 'var(--fg)' }}>
                  {cls}
                </div>
                <div style={{ flex: 1, height: 16, background: 'var(--bg-card-inner, var(--border))',
                              borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${(count / maxCount) * 100}%`,
                    background: cls === 'BENIGN' ? 'var(--color-green, #22c55e)' : 'var(--color-blue, #3b82f6)',
                    borderRadius: 3,
                    minWidth: 2,
                  }} />
                </div>
                <div style={{ width: 90, fontSize: 12, color: 'var(--fg-muted)', flexShrink: 0 }}>
                  {count.toLocaleString()} ({((count / totalRows) * 100).toFixed(1)}%)
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function splitLabel(fname: string): string {
  const n = fname.toLowerCase()
  if (!n.includes('friday')) return 'Train (Mon–Thu)'
  if (n.includes('morning') || n.includes('workinghours')) return 'Val (Fri morning)'
  return 'Test (Fri afternoon)'
}
