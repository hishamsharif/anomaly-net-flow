import { useEffect, useRef, useState } from 'react'

interface Props { hasModel: boolean | null }

interface ModelInfo {
  id: number
  classifier_type: string
  detector_type: string
  status: string
  metrics: Record<string, number | string | null>
  created_at: string
}

interface ModelHistory {
  id: number
  classifier_type: string
  detector_type: string
  status: string
  metrics: Record<string, number | string | null>
  created_at: string
}

export default function TrainingPage({ hasModel }: Props) {
  const [activeModel, setActiveModel] = useState<ModelInfo | null>(null)
  const [history, setHistory] = useState<ModelHistory[]>([])
  const [training, setTraining] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [verdictCount, setVerdictCount] = useState(0)
  const K = 50
  const logRef = useRef<HTMLDivElement>(null)

  const fetchModel = () =>
    fetch('/api/models/active')
      .then(r => r.json())
      .then(d => setActiveModel(d.model ?? null))
      .catch(() => {})

  const fetchHistory = () =>
    fetch('/api/models/')
      .then(r => r.json())
      .then(d => setHistory(d.models ?? []))
      .catch(() => {})

  const fetchVerdicts = () =>
    fetch('/api/alerts/metrics')
      .then(r => r.json())
      .then(d => setVerdictCount(d.verdict_count ?? 0))
      .catch(() => {})

  useEffect(() => {
    fetchModel()
    fetchHistory()
    fetchVerdicts()
  }, [hasModel])

  useEffect(() => {
    if (logRef.current)
      logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  const startTraining = async (mode: 'initial' | 'fine_tune') => {
    setTraining(true)
    setLogs([`Starting ${mode} training…`])
    try {
      const resp = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        setLogs(l => [...l, `Error: ${err.detail ?? JSON.stringify(err)}`])
        setTraining(false)
        return
      }
      const { run_id } = await resp.json()
      const es = new EventSource(`/api/training/events/${run_id}`)
      es.addEventListener('progress', e => setLogs(l => [...l, e.data]))
      es.addEventListener('done', e => {
        setLogs(l => [...l, `✔ ${e.data}`])
        setTraining(false)
        es.close()
        fetchModel()
        fetchHistory()
        fetchVerdicts()
      })
      es.addEventListener('error', e => {
        const msg = (e as MessageEvent).data ?? 'Unknown error'
        setLogs(l => [...l, `✘ ${msg}`])
        setTraining(false)
        es.close()
      })
      es.onerror = () => {
        setLogs(l => [...l, '(connection closed)'])
        setTraining(false)
        es.close()
      }
    } catch (err) {
      setLogs(l => [...l, `Error: ${String(err)}`])
      setTraining(false)
    }
  }

  const fmt = (v: number | string | null | undefined, decimals = 4) =>
    v == null ? '—' : typeof v === 'number' ? v.toFixed(decimals) : String(v)

  return (
    <div className="page">
      <h1>Train &amp; Evaluate</h1>

      {/* ── No model yet ── */}
      {!activeModel && (
        <div className="card">
          <p>No trained model found. Train on the bundled CIC-IDS2017 dataset to get started.</p>
          <p>Make sure the CSV files are present in <code>data/CIC-IDS2017/</code> before starting.</p>
          <br />
          <button
            className="btn btn-primary"
            disabled={training}
            onClick={() => startTraining('initial')}
          >
            {training ? 'Training…' : 'Train on CIC-IDS2017'}
          </button>
        </div>
      )}

      {/* ── Active model metrics ── */}
      {activeModel && (
        <>
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <strong>Active model v{activeModel.id}</strong>{' '}
                <span className="badge badge-green">active</span>
              </div>
              <small style={{ color: 'var(--fg-muted)' }}>
                {activeModel.classifier_type} + {activeModel.detector_type}
              </small>
            </div>
            <div className="metrics">
              <div className="metric">
                <div className="metric-label">F1 (macro)</div>
                <div className="metric-value">{fmt(activeModel.metrics?.f1)}</div>
              </div>
              <div className="metric">
                <div className="metric-label">PR-AUC</div>
                <div className="metric-value">{fmt(activeModel.metrics?.pr_auc)}</div>
              </div>
              <div className="metric">
                <div className="metric-label">Recall</div>
                <div className="metric-value">{fmt(activeModel.metrics?.recall)}</div>
              </div>
              <div className="metric">
                <div className="metric-label">FAR</div>
                <div className="metric-value">{fmt(activeModel.metrics?.far)}</div>
              </div>
              <div className="metric">
                <div className="metric-label">AMR</div>
                <div className="metric-value">{fmt(activeModel.metrics?.amr)}</div>
              </div>
              <div className="metric">
                <div className="metric-label">Train rows</div>
                <div className="metric-value">{fmt(activeModel.metrics?.train_rows, 0)}</div>
              </div>
            </div>
          </div>

          {/* Fine-tune card */}
          <div className="card">
            <strong>Incremental fine-tune</strong>
            <p>
              Analyst verdicts since last training:{' '}
              <strong>{verdictCount}</strong> / {K}
            </p>
            <div className="progress-wrap">
              <div
                className="progress-bar"
                style={{ width: `${Math.min((verdictCount / K) * 100, 100)}%` }}
              />
            </div>
            {verdictCount < K && (
              <p style={{ fontSize: 13, marginTop: 4 }}>
                Submit {K - verdictCount} more analyst verdict{K - verdictCount !== 1 ? 's' : ''} on
                the Inference tab to unlock fine-tuning.
              </p>
            )}
            <br />
            <button
              className="btn btn-primary"
              disabled={training || verdictCount < K}
              onClick={() => startTraining('fine_tune')}
            >
              {training ? 'Fine-tuning…' : 'Fine-tune on analyst verdicts'}
            </button>
          </div>
        </>
      )}

      {/* ── Training log ── */}
      {logs.length > 0 && (
        <>
          <h2>Training log</h2>
          <div className="log" ref={logRef}>
            {logs.join('\n')}
          </div>
        </>
      )}

      {/* ── Model history ── */}
      {history.length > 0 && (
        <>
          <h2>Model history</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Type</th>
                  <th>F1</th>
                  <th>PR-AUC</th>
                  <th>Recall</th>
                  <th>FAR</th>
                  <th>AMR</th>
                  <th>Status</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {history.map(m => (
                  <tr key={m.id}>
                    <td>v{m.id}</td>
                    <td style={{ fontSize: 12 }}>{m.classifier_type} + {m.detector_type}</td>
                    <td>{fmt(m.metrics?.f1)}</td>
                    <td>{fmt(m.metrics?.pr_auc)}</td>
                    <td>{fmt(m.metrics?.recall)}</td>
                    <td>{fmt(m.metrics?.far)}</td>
                    <td>{fmt(m.metrics?.amr)}</td>
                    <td>
                      <span className={`badge ${m.status === 'active' ? 'badge-green' : m.status === 'failed' ? 'badge-red' : 'badge-yellow'}`}>
                        {m.status}
                      </span>
                    </td>
                    <td style={{ fontSize: 12 }}>{new Date(m.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
