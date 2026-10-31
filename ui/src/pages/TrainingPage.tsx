import { useEffect, useRef, useState } from 'react'
import DatasetPanel from './DatasetPage'
import PreprocessPanel from './PreprocessPage'
import EDAPanel from './EDAPage'
import FeaturesPanel from './FeaturesPage'

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

const CLF_LABELS: Record<string, string> = {
  xgboost: 'XGBoost',
  random_forest: 'Random Forest',
}
const DET_LABELS: Record<string, string> = {
  mlp_ae: 'MLP Autoencoder',
  isolation_forest: 'Isolation Forest',
}
const CLF_SUBTITLE = 'Supervised — trained with labelled data to classify known attack categories'
const DET_SUBTITLE = 'Unsupervised — trained on normal traffic only; flags deviations (catches zero-days)'

const label = (map: Record<string, string>, key: string) =>
  map[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

type SubTab = 'dataset' | 'preprocess' | 'eda' | 'features' | 'train'

const SUB_TABS: { id: SubTab; label: string }[] = [
  { id: 'dataset',    label: 'Dataset' },
  { id: 'preprocess', label: 'Preprocessing' },
  { id: 'eda',        label: 'EDA' },
  { id: 'features',   label: 'Features' },
  { id: 'train',      label: 'Train & Evaluate' },
]

export default function TrainingPage({ hasModel }: Props) {
  const [subTab, setSubTab] = useState<SubTab>('dataset')
  const [activeModel, setActiveModel] = useState<ModelInfo | null>(null)
  const [history, setHistory] = useState<ModelInfo[]>([])
  const [training, setTraining] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [pct, setPct] = useState<number | null>(null)
  const [verdictCount, setVerdictCount] = useState(0)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const [classifiers, setClassifiers] = useState<string[]>(['xgboost'])
  const [detectors, setDetectors] = useState<string[]>(['mlp_ae'])
  const [clf, setClf] = useState('xgboost')
  const [det, setDet] = useState('mlp_ae')

  const K = 50
  const logRef = useRef<HTMLDivElement>(null)

  const fetchModel = () =>
    fetch('/api/models/active').then(r => r.json()).then(d => setActiveModel(d.model ?? null)).catch(() => {})
  const fetchHistory = () =>
    fetch('/api/models/').then(r => r.json()).then(d => setHistory(d.models ?? [])).catch(() => {})
  const fetchVerdicts = () =>
    fetch('/api/alerts/metrics').then(r => r.json()).then(d => setVerdictCount(d.verdict_count ?? 0)).catch(() => {})
  const fetchTypes = () =>
    fetch('/api/training/available-types').then(r => r.json()).then(d => {
      if (d.classifiers?.length) setClassifiers(d.classifiers)
      if (d.detectors?.length) setDetectors(d.detectors)
    }).catch(() => {})

  useEffect(() => {
    fetchModel(); fetchHistory(); fetchVerdicts(); fetchTypes()
  }, [hasModel])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  const stopPolling = () => {
    if (pollRef.current !== null) { clearInterval(pollRef.current); pollRef.current = null }
  }
  const startPolling = (rid: string) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`/api/training/status/${rid}`)
        if (!r.ok) return
        const s = await r.json()
        if (s.pct !== undefined) setPct(Number(s.pct))
        if (Number(s.pct) >= 100) { stopPolling(); setTraining(false); fetchModel(); fetchHistory(); fetchVerdicts() }
      } catch {}
    }, 5000)
  }

  const startTraining = async (mode: 'initial' | 'fine_tune') => {
    setTraining(true); setPct(0)
    setLogs([`Starting ${mode === 'initial' ? 'initial' : 'fine-tune'} training…`])
    try {
      const resp = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, classifier_type: clf, detector_type: det }),
      })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        setLogs(l => [...l, `Error: ${err.detail ?? JSON.stringify(err)}`]); setTraining(false); return
      }
      const { run_id, estimated_label } = await resp.json()
      setLogs(l => [...l, `Acknowledged — estimated completion: ${estimated_label}`])
      const es = new EventSource(`/api/training/events/${run_id}`)
      es.addEventListener('step', e => setPct(Number(e.data)))
      es.addEventListener('progress', e => setLogs(l => [...l, e.data]))
      es.addEventListener('done', e => {
        stopPolling(); setLogs(l => [...l, `✔ ${e.data}`]); setPct(100); setTraining(false); es.close()
        fetchModel(); fetchHistory(); fetchVerdicts()
      })
      es.addEventListener('error', e => {
        stopPolling(); setLogs(l => [...l, `✘ ${(e as MessageEvent).data ?? 'Unknown error'}`])
        setPct(null); setTraining(false); es.close()
      })
      es.onerror = () => {
        es.close(); setLogs(l => [...l, '(SSE disconnected — polling every 5 s…)']); startPolling(run_id)
      }
    } catch (err) { setLogs(l => [...l, `Error: ${String(err)}`]); setTraining(false) }
  }

  const fmt = (v: number | string | null | undefined, decimals = 4) =>
    v == null ? '—' : typeof v === 'number' ? v.toFixed(decimals) : String(v)

  const ModelSelector = () => (
    <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', margin: '12px 0' }}>
      {[
        { title: 'Stage 2 — Known-Attack Classifier', subtitle: CLF_SUBTITLE, value: clf, set: setClf, opts: classifiers, map: CLF_LABELS },
        { title: 'Stage 3 — Anomaly Detector', subtitle: DET_SUBTITLE, value: det, set: setDet, opts: detectors, map: DET_LABELS },
      ].map(({ title, subtitle, value, set, opts, map }) => (
        <div key={title} style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--fg)', marginBottom: 2 }}>{title}</div>
          <div style={{ fontSize: 11, color: 'var(--fg-muted)', marginBottom: 6, lineHeight: 1.4 }}>{subtitle}</div>
          <select value={value} onChange={e => set(e.target.value)} disabled={training}
            style={{ width: '100%', padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)', color: 'var(--fg)' }}>
            {opts.map(o => <option key={o} value={o}>{label(map, o)}</option>)}
          </select>
        </div>
      ))}
    </div>
  )

  return (
    <div className="page">
      <h1>Training Pipeline</h1>

      {/* ── Sub-tab navigation ── */}
      <div style={{
        display: 'flex', gap: 0, borderBottom: '2px solid var(--border)',
        marginBottom: 24, overflowX: 'auto',
      }}>
        {SUB_TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setSubTab(t.id)}
            style={{
              padding: '8px 18px', border: 'none', background: 'none', cursor: 'pointer',
              fontSize: 14, fontWeight: subTab === t.id ? 600 : 400,
              color: subTab === t.id ? 'var(--fg)' : 'var(--fg-muted)',
              borderBottom: subTab === t.id ? '2px solid var(--fg)' : '2px solid transparent',
              marginBottom: -2, whiteSpace: 'nowrap',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Sub-tab content ── */}
      {subTab === 'dataset'    && <DatasetPanel />}
      {subTab === 'preprocess' && <PreprocessPanel />}
      {subTab === 'eda'        && <EDAPanel />}
      {subTab === 'features'   && <FeaturesPanel />}

      {subTab === 'train' && (
        <>
          {/* ── No model: initial training ── */}
          {!activeModel && (
            <div className="card">
              <h2 style={{ marginTop: 0 }}>Phase 1 — Initial Training</h2>
              <p>
                No trained model yet. Select a classifier and anomaly detector,
                then train on CIC-IDS2017 (2.8 M flows, 15 attack classes).
              </p>
              <p style={{ fontSize: 13, color: 'var(--fg-muted)' }}>
                Make sure CSV files are in <code>data/CIC-IDS2017/</code> before starting.
              </p>
              <ModelSelector />
              <button className="btn btn-primary" disabled={training} onClick={() => startTraining('initial')}>
                {training ? 'Training…' : `Train ${label(CLF_LABELS, clf)} + ${label(DET_LABELS, det)}`}
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
                      {activeModel.name ?? `${activeModel.classifier_type} + ${activeModel.detector_type} v${activeModel.id}`}
                    </div>
                    <span className="badge badge-green">active</span>
                  </div>
                  <small style={{ color: 'var(--fg-muted)', alignSelf: 'center' }}>
                    {new Date(activeModel.created_at).toLocaleString()}
                  </small>
                </div>
                <div className="metrics" style={{ marginTop: 12 }}>
                  {[
                    ['F1 — known attacks', activeModel.metrics?.known_attack_f1 ?? activeModel.metrics?.f1],
                    ['F1 (all classes)',   activeModel.metrics?.f1],
                    ['PR-AUC',            activeModel.metrics?.pr_auc],
                    ['Recall',            activeModel.metrics?.recall],
                    ['FAR',               activeModel.metrics?.far],
                    ['AMR',               activeModel.metrics?.amr],
                    ['AE threshold',      activeModel.metrics?.ae_threshold],
                    ['Train rows',        activeModel.metrics?.train_rows],
                  ].map(([lbl, val]) => (
                    <div key={String(lbl)} className="metric">
                      <div className="metric-label">{lbl}</div>
                      <div className="metric-value">{fmt(val as number | null)}</div>
                    </div>
                  ))}
                </div>
              </div>

              <details className="card" style={{ cursor: 'pointer' }}>
                <summary style={{ fontWeight: 600, userSelect: 'none' }}>Train a different model pair</summary>
                <p style={{ marginTop: 8, fontSize: 13, color: 'var(--fg-muted)' }}>
                  Retrains on the full CIC-IDS2017 dataset. The new model becomes active; the current one is archived.
                </p>
                <ModelSelector />
                <button className="btn btn-primary" disabled={training} onClick={() => startTraining('initial')}>
                  {training ? 'Training…' : `Train ${label(CLF_LABELS, clf)} + ${label(DET_LABELS, det)}`}
                </button>
              </details>

              <div className="card">
                <strong>Incremental fine-tune</strong>
                <p style={{ marginTop: 4 }}>
                  Analyst verdicts since last training: <strong>{verdictCount}</strong> / {K}
                </p>
                <div className="progress-wrap">
                  <div className="progress-bar" style={{ width: `${Math.min((verdictCount / K) * 100, 100)}%` }} />
                </div>
                {verdictCount < K && (
                  <p style={{ fontSize: 13, marginTop: 6, color: 'var(--fg-muted)' }}>
                    Submit {K - verdictCount} more verdict{K - verdictCount !== 1 ? 's' : ''} on the Inference tab to unlock fine-tuning.
                  </p>
                )}
                <br />
                <button className="btn btn-primary" disabled={training || verdictCount < K} onClick={() => startTraining('fine_tune')}>
                  {training ? 'Fine-tuning…' : 'Fine-tune on analyst verdicts'}
                </button>
              </div>
            </>
          )}

          {/* ── Training log ── */}
          {logs.length > 0 && (
            <>
              <h2>Training log</h2>
              {pct !== null && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: 13 }}>
                    <span style={{ color: 'var(--fg-muted)' }}>{pct < 100 ? 'Training in progress…' : 'Complete'}</span>
                    <span style={{ fontWeight: 600 }}>{pct}%</span>
                  </div>
                  <div className="progress-wrap">
                    <div className="progress-bar" style={{
                      width: `${pct}%`, transition: 'width 0.4s ease',
                      background: pct === 100 ? 'var(--color-green, #22c55e)' : undefined,
                    }} />
                  </div>
                </div>
              )}
              <div className="log" ref={logRef}>{logs.join('\n')}</div>
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
                      <th>Name</th><th>F1 (known)</th><th>F1 (macro)</th>
                      <th>FAR</th><th>AMR</th><th>AE threshold</th><th>Status</th><th>Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map(m => (
                      <tr key={m.id}>
                        <td style={{ fontSize: 13 }}>
                          {m.name ?? `${m.classifier_type} + ${m.detector_type} v${m.id}`}
                        </td>
                        <td>{fmt(m.metrics?.known_attack_f1 ?? m.metrics?.f1)}</td>
                        <td>{fmt(m.metrics?.f1)}</td>
                        <td>{fmt(m.metrics?.far)}</td>
                        <td>{fmt(m.metrics?.amr)}</td>
                        <td>{fmt(m.metrics?.ae_threshold)}</td>
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
        </>
      )}
    </div>
  )
}
