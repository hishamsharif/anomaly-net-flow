import { useEffect, useRef, useState } from 'react'

interface Props { hasModel: boolean | null }

interface ModelInfo {
  id: number
  name: string | null
  classifier_type: string
  detector_type: string
  status: string
  metrics: Record<string, number | string | null>
  created_at: string
}

// Human-readable labels for registry keys
const CLF_LABELS: Record<string, string> = {
  xgboost: 'XGBoost',
  random_forest: 'Random Forest',
}
const DET_LABELS: Record<string, string> = {
  mlp_ae: 'MLP Autoencoder',
  isolation_forest: 'Isolation Forest',
}

// Section subtitles explaining the role of each stage
const CLF_SUBTITLE = 'Supervised — trained with labelled data to classify known attack categories (DoS, PortScan, etc.)'
const DET_SUBTITLE = 'Unsupervised — trained on normal traffic only; flags anything that deviates from the baseline (catches zero-days)'
const label = (map: Record<string, string>, key: string) =>
  map[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

export default function TrainingPage({ hasModel }: Props) {
  const [activeModel, setActiveModel] = useState<ModelInfo | null>(null)
  const [history, setHistory] = useState<ModelInfo[]>([])
  const [training, setTraining] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [pct, setPct] = useState<number | null>(null)
  const [verdictCount, setVerdictCount] = useState(0)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Available types loaded from the API
  const [classifiers, setClassifiers] = useState<string[]>(['xgboost'])
  const [detectors, setDetectors] = useState<string[]>(['mlp_ae'])

  // Phase 1 selections
  const [clf, setClf] = useState('xgboost')
  const [det, setDet] = useState('mlp_ae')

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

  const fetchTypes = () =>
    fetch('/api/training/available-types')
      .then(r => r.json())
      .then(d => {
        if (d.classifiers?.length) setClassifiers(d.classifiers)
        if (d.detectors?.length) setDetectors(d.detectors)
      })
      .catch(() => {})

  useEffect(() => {
    fetchModel()
    fetchHistory()
    fetchVerdicts()
    fetchTypes()
  }, [hasModel])

  useEffect(() => {
    if (logRef.current)
      logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  const stopPolling = () => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const startPolling = (rid: string) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`/api/training/status/${rid}`)
        if (!r.ok) return
        const s = await r.json()
        if (s.pct !== undefined) setPct(Number(s.pct))
        if (Number(s.pct) >= 100) {
          stopPolling()
          setTraining(false)
          fetchModel()
          fetchHistory()
          fetchVerdicts()
        }
      } catch {}
    }, 5000)
  }

  const startTraining = async (mode: 'initial' | 'fine_tune') => {
    setTraining(true)
    setPct(0)
    setLogs([`Starting ${mode === 'initial' ? 'initial' : 'fine-tune'} training…`])
    try {
      const resp = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, classifier_type: clf, detector_type: det }),
      })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        setLogs(l => [...l, `Error: ${err.detail ?? JSON.stringify(err)}`])
        setTraining(false)
        return
      }
      const { run_id, estimated_label } = await resp.json()
      setLogs(l => [...l, `Acknowledged — estimated completion: ${estimated_label}`])
      const es = new EventSource(`/api/training/events/${run_id}`)
      es.addEventListener('step', e => setPct(Number(e.data)))
      es.addEventListener('progress', e => setLogs(l => [...l, e.data]))
      es.addEventListener('done', e => {
        stopPolling()
        setLogs(l => [...l, `✔ ${e.data}`])
        setPct(100)
        setTraining(false)
        es.close()
        fetchModel()
        fetchHistory()
        fetchVerdicts()
      })
      es.addEventListener('error', e => {
        const msg = (e as MessageEvent).data ?? 'Unknown error'
        stopPolling()
        setLogs(l => [...l, `✘ ${msg}`])
        setPct(null)
        setTraining(false)
        es.close()
      })
      es.onerror = () => {
        es.close()
        setLogs(l => [...l, '(SSE disconnected — polling for progress every 5 s…)'])
        startPolling(run_id)
      }
    } catch (err) {
      setLogs(l => [...l, `Error: ${String(err)}`])
      setTraining(false)
    }
  }

  const fmt = (v: number | string | null | undefined, decimals = 4) =>
    v == null ? '—' : typeof v === 'number' ? v.toFixed(decimals) : String(v)

  // Model selector UI — shared between "no model" and "train new model" flows
  const ModelSelector = () => (
    <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', margin: '12px 0' }}>
      <div style={{ flex: 1, minWidth: 200 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--fg)', marginBottom: 2 }}>
          Stage 2 — Known-Attack Classifier
        </div>
        <div style={{ fontSize: 11, color: 'var(--fg-muted)', marginBottom: 6, lineHeight: 1.4 }}>
          {CLF_SUBTITLE}
        </div>
        <select
          value={clf}
          onChange={e => setClf(e.target.value)}
          disabled={training}
          style={{ width: '100%', padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)', color: 'var(--fg)' }}
        >
          {classifiers.map(c => (
            <option key={c} value={c}>{label(CLF_LABELS, c)}</option>
          ))}
        </select>
      </div>
      <div style={{ flex: 1, minWidth: 200 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--fg)', marginBottom: 2 }}>
          Stage 3 — Anomaly Detector
        </div>
        <div style={{ fontSize: 11, color: 'var(--fg-muted)', marginBottom: 6, lineHeight: 1.4 }}>
          {DET_SUBTITLE}
        </div>
        <select
          value={det}
          onChange={e => setDet(e.target.value)}
          disabled={training}
          style={{ width: '100%', padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)', color: 'var(--fg)' }}
        >
          {detectors.map(d => (
            <option key={d} value={d}>{label(DET_LABELS, d)}</option>
          ))}
        </select>
      </div>
    </div>
  )

  return (
    <div className="page">
      <h1>Train &amp; Evaluate</h1>

      {/* ── Phase 1: No model — initial training ── */}
      {!activeModel && (
        <div className="card">
          <h2 style={{ marginTop: 0 }}>Phase 1 — Initial Training</h2>
          <p>
            No trained model yet. Select a classifier and anomaly detector,
            then train on the bundled CIC-IDS2017 dataset (2.8 M flows, 15 attack classes).
          </p>
          <p style={{ fontSize: 13, color: 'var(--fg-muted)' }}>
            Make sure CSV files are present in <code>data/CIC-IDS2017/</code> before starting.
          </p>
          <ModelSelector />
          <button
            className="btn btn-primary"
            disabled={training}
            onClick={() => startTraining('initial')}
          >
            {training ? 'Training…' : `Train ${label(CLF_LABELS, clf)} (classifier) + ${label(DET_LABELS, det)} (detector)`}
          </button>
        </div>
      )}

      {/* ── Active model metrics ── */}
      {activeModel && (
        <>
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>
                  {activeModel.name ?? `Classifier:${activeModel.classifier_type} | Detector:${activeModel.detector_type} | v${activeModel.id}`}
                </div>
                <span className="badge badge-green">active</span>
              </div>
              <small style={{ color: 'var(--fg-muted)', alignSelf: 'center' }}>
                {new Date(activeModel.created_at).toLocaleString()}
              </small>
            </div>
            <div className="metrics" style={{ marginTop: 12 }}>
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

          {/* Train a new model (different pair) */}
          <details className="card" style={{ cursor: 'pointer' }}>
            <summary style={{ fontWeight: 600, userSelect: 'none' }}>
              Train a different model pair
            </summary>
            <p style={{ marginTop: 8, fontSize: 13, color: 'var(--fg-muted)' }}>
              Train a new classifier + detector on the full CIC-IDS2017 dataset.
              The new model will become active and the current one will be archived.
            </p>
            <ModelSelector />
            <button
              className="btn btn-primary"
              disabled={training}
              onClick={() => startTraining('initial')}
            >
              {training ? 'Training…' : `Train ${label(CLF_LABELS, clf)} (classifier) + ${label(DET_LABELS, det)} (detector)`}
            </button>
          </details>

          {/* Fine-tune card */}
          <div className="card">
            <strong>Incremental fine-tune</strong>
            <p style={{ marginTop: 4 }}>
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
              <p style={{ fontSize: 13, marginTop: 6, color: 'var(--fg-muted)' }}>
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

      {/* ── Training progress + log ── */}
      {logs.length > 0 && (
        <>
          <h2>Training log</h2>

          {/* Progress bar — visible while training or just completed */}
          {pct !== null && (
            <div style={{ marginBottom: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: 13 }}>
                <span style={{ color: 'var(--fg-muted)' }}>
                  {pct < 100 ? 'Training in progress…' : 'Complete'}
                </span>
                <span style={{ fontWeight: 600 }}>{pct}%</span>
              </div>
              <div className="progress-wrap">
                <div
                  className="progress-bar"
                  style={{
                    width: `${pct}%`,
                    transition: 'width 0.4s ease',
                    background: pct === 100 ? 'var(--color-green, #22c55e)' : undefined,
                  }}
                />
              </div>
            </div>
          )}

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
                  <th>Name</th>
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
                    <td style={{ fontSize: 13 }}>
                      {m.name ?? `Classifier:${m.classifier_type} | Detector:${m.detector_type} | v${m.id}`}
                    </td>
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
