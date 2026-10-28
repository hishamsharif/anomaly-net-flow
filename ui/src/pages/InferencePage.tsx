import { useRef, useState } from 'react'

interface Props { hasModel: boolean | null }

interface FlowResult {
  flow_id: string
  rule_verdict: string
  predicted_label: string | null
  classifier_confidence: number | null
  ae_score: number | null
  consensus: string
  severity: string
  alert_id: number
}

type Verdict = 'analyst_confirmed' | 'analyst_override'

export default function InferencePage({ hasModel }: Props) {
  const [results, setResults] = useState<FlowResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [overrides, setOverrides] = useState<Record<number, { verdict: Verdict; label: string }>>({})
  const [submitted, setSubmitted] = useState<Set<number>>(new Set())
  const fileRef = useRef<HTMLInputElement>(null)

  if (hasModel === false) {
    return (
      <div className="page">
        <h1>Inference</h1>
        <div className="card">
          <p>No trained model is available yet.</p>
          <p>Go to the <strong>Train &amp; Evaluate</strong> tab and train on CIC-IDS2017 first.</p>
        </div>
      </div>
    )
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError(null)
    setLoading(true)
    setResults([])
    setOverrides({})
    setSubmitted(new Set())
    try {
      const form = new FormData()
      form.append('file', file)
      const resp = await fetch('/api/predict/batch', { method: 'POST', body: form })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(err.detail ?? JSON.stringify(err))
      }
      const data = await resp.json()
      setResults(data.results ?? [])
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  const submitFeedback = async (alertId: number) => {
    const fb = overrides[alertId]
    if (!fb) return
    try {
      const resp = await fetch(`/api/alerts/${alertId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ verdict: fb.verdict, correct_label: fb.label }),
      })
      if (!resp.ok) throw new Error('Failed to submit feedback')
      setSubmitted(s => new Set([...s, alertId]))
    } catch (err) {
      alert(String(err))
    }
  }

  const severityBadge = (s: string) => {
    if (s === 'critical') return 'badge-red'
    if (s === 'medium') return 'badge-yellow'
    return 'badge-green'
  }

  return (
    <div className="page">
      <h1>Inference</h1>
      <p>
        Upload a CSV (CIC-IDS2017 format, Label column optional) or a PCAP file.
        Each flow is classified through the 3-stage pipeline.
      </p>

      {/* ── Upload ── */}
      <div
        className="upload-zone"
        onClick={() => fileRef.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={e => {
          e.preventDefault()
          const file = e.dataTransfer.files?.[0]
          if (file && fileRef.current) {
            const dt = new DataTransfer()
            dt.items.add(file)
            fileRef.current.files = dt.files
            fileRef.current.dispatchEvent(new Event('change', { bubbles: true }))
          }
        }}
      >
        {loading ? 'Running inference…' : 'Click or drag a CSV / PCAP file here'}
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.pcap,.pcapng"
          style={{ display: 'none' }}
          onChange={handleUpload}
        />
      </div>

      {error && (
        <div className="card" style={{ borderColor: 'var(--danger)', marginTop: 16 }}>
          <strong style={{ color: 'var(--danger)' }}>Error:</strong> {error}
        </div>
      )}

      {/* ── Results ── */}
      {results.length > 0 && (
        <>
          <h2>Results — {results.length} flows</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Flow</th>
                  <th>Rule</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                  <th>AE Score</th>
                  <th>Severity</th>
                  <th>Analyst verdict</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {results.map(r => (
                  <tr key={r.alert_id} style={submitted.has(r.alert_id) ? { opacity: 0.5 } : {}}>
                    <td style={{ fontSize: 12, maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {r.flow_id}
                    </td>
                    <td>
                      <span className={`badge ${r.rule_verdict === 'BENIGN' ? 'badge-green' : 'badge-yellow'}`}>
                        {r.rule_verdict}
                      </span>
                    </td>
                    <td>
                      <strong>{r.predicted_label ?? '—'}</strong>
                    </td>
                    <td>{r.classifier_confidence != null ? `${(r.classifier_confidence * 100).toFixed(1)}%` : '—'}</td>
                    <td>{r.ae_score != null ? r.ae_score.toFixed(4) : '—'}</td>
                    <td>
                      <span className={`badge ${severityBadge(r.severity)}`}>
                        {r.severity}
                      </span>
                    </td>
                    <td>
                      {submitted.has(r.alert_id) ? (
                        <span className="badge badge-green">submitted</span>
                      ) : (
                        <select
                          style={{ fontSize: 13, padding: '4px 8px', borderRadius: 4 }}
                          value={overrides[r.alert_id]?.verdict ?? ''}
                          onChange={e => {
                            const v = e.target.value as Verdict
                            setOverrides(o => ({
                              ...o,
                              [r.alert_id]: {
                                verdict: v,
                                label: v === 'analyst_confirmed'
                                  ? (r.predicted_label ?? 'BENIGN')
                                  : o[r.alert_id]?.label ?? '',
                              },
                            }))
                          }}
                        >
                          <option value="">— review —</option>
                          <option value="analyst_confirmed">✔ Confirm prediction</option>
                          <option value="analyst_override">✎ Override label</option>
                        </select>
                      )}
                      {overrides[r.alert_id]?.verdict === 'analyst_override' && !submitted.has(r.alert_id) && (
                        <input
                          style={{ marginLeft: 6, fontSize: 13, padding: '4px 8px', borderRadius: 4, width: 120 }}
                          placeholder="correct label"
                          value={overrides[r.alert_id]?.label ?? ''}
                          onChange={e =>
                            setOverrides(o => ({
                              ...o,
                              [r.alert_id]: { ...o[r.alert_id], label: e.target.value },
                            }))
                          }
                        />
                      )}
                    </td>
                    <td>
                      {!submitted.has(r.alert_id) && overrides[r.alert_id]?.verdict && (
                        <button
                          className="btn btn-outline"
                          style={{ padding: '4px 12px', fontSize: 13 }}
                          onClick={() => submitFeedback(r.alert_id)}
                        >
                          Submit
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ marginTop: 12, fontSize: 13 }}>
            Submitted verdicts count toward the fine-tune threshold on the Train &amp; Evaluate tab.
          </p>
        </>
      )}
    </div>
  )
}
