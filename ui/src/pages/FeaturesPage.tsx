import { useEffect, useState } from 'react'

interface TopFeature { name: string; importance: number }
interface Stats {
  n_features_initial: number | null
  n_features_after_vt: number | null
  n_features_after_corr: number | null
  n_features_final: number | null
  variance_threshold: number | null
  correlation_threshold: number | null
  feature_selection_k: number | null
  clf_scaler_type: string | null
  det_scaler_type: string | null
  top_features: TopFeature[]
  selected_feature_names: string[]
}

export default function FeaturesPanel() {
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

  if (loading) return <div><h2>Feature Engineering</h2><p>Loading…</p></div>

  if (modelId == null) {
    return (
      <div>
        <h2>Feature Engineering</h2>
        <div className="card">
          <p style={{ color: 'var(--fg-muted)', margin: 0 }}>
            No trained model yet — feature engineering results are computed during initial training.
            Go to the <strong>Train &amp; Evaluate</strong> sub-tab to start.
          </p>
        </div>
      </div>
    )
  }

  const topFeatures = stats?.top_features ?? []
  const maxImportance = topFeatures[0]?.importance ?? 1

  const stages = [
    { label: 'Raw features', count: stats?.n_features_initial, note: 'All CICFlowMeter columns' },
    {
      label: 'After VarianceThreshold',
      count: stats?.n_features_after_vt,
      note: `threshold=${stats?.variance_threshold ?? 0} — drops near-constant columns`,
    },
    {
      label: 'After correlation pruning',
      count: stats?.n_features_after_corr,
      note: `|r| > ${stats?.correlation_threshold ?? 0.95} — drops one from each correlated pair`,
    },
    {
      label: 'Final (after MI selection)',
      count: stats?.n_features_final,
      note: stats?.feature_selection_k
        ? `SelectKBest(k=${stats.feature_selection_k}) — top-K by mutual information`
        : 'MI selection skipped (FEATURE_SELECTION_K=0)',
    },
  ]

  return (
    <div>
      <h2>Feature Engineering</h2>
      <p>
        Four-stage pipeline: variance filtering removes near-constant features,
        correlation pruning removes redundancy, optional mutual-information selection
        keeps the top-K most informative features, then path-specific scalers are applied.
        All stages are fitted on training data only.
      </p>

      {/* ── Funnel ── */}
      <h3>Feature count funnel</h3>
      <div className="card">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {stages.map((s, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
              <div style={{
                width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
                background: i === stages.length - 1 ? 'var(--color-green, #22c55e)' : 'var(--color-blue, #3b82f6)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: '#fff', fontWeight: 700, fontSize: 13,
              }}>
                {i + 1}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 600, fontSize: 14 }}>{s.label}</span>
                  <span style={{
                    fontWeight: 700, fontSize: 20,
                    color: i === stages.length - 1 ? 'var(--color-green, #22c55e)' : 'var(--fg)',
                  }}>
                    {s.count != null ? s.count : '—'}
                  </span>
                </div>
                <div style={{ fontSize: 12, color: 'var(--fg-muted)', marginTop: 2 }}>{s.note}</div>
                {i < stages.length - 1 && s.count != null && stages[i + 1].count != null && (
                  <div style={{ fontSize: 12, color: 'var(--fg-muted)', marginTop: 2 }}>
                    → dropped {s.count! - stages[i + 1].count!} features
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Top features ── */}
      {topFeatures.length > 0 && (
        <>
          <h3>Top-15 features by classifier importance</h3>
          <div className="card">
            <p style={{ marginTop: 0, fontSize: 13, color: 'var(--fg-muted)' }}>
              Ranked by XGBoost feature importance (gain). Higher = more decision power.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {topFeatures.map((f, i) => (
                <div key={f.name} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ width: 20, fontSize: 12, color: 'var(--fg-muted)', textAlign: 'right',
                                flexShrink: 0 }}>
                    {i + 1}
                  </div>
                  <div style={{ width: 240, fontSize: 12, flexShrink: 0, overflow: 'hidden',
                                textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {f.name}
                  </div>
                  <div style={{ flex: 1, height: 14, background: 'var(--bg-card-inner, var(--border))',
                                borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: `${(f.importance / maxImportance) * 100}%`,
                      background: 'var(--color-blue, #3b82f6)',
                      borderRadius: 3,
                      minWidth: 2,
                    }} />
                  </div>
                  <div style={{ width: 60, fontSize: 12, color: 'var(--fg-muted)', flexShrink: 0,
                                textAlign: 'right' }}>
                    {f.importance.toFixed(4)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {topFeatures.length === 0 && (stats?.selected_feature_names?.length ?? 0) > 0 && (
        <>
          <h3>Selected features ({stats!.selected_feature_names.length})</h3>
          <div className="card">
            <p style={{ marginTop: 0, fontSize: 13, color: 'var(--fg-muted)' }}>
              Feature importance not available for the active detector type.
              Showing selected feature names.
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {stats!.selected_feature_names.map(n => (
                <code key={n} style={{ fontSize: 12, padding: '2px 6px', borderRadius: 4,
                                       background: 'var(--bg-card-inner, var(--border))' }}>
                  {n}
                </code>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
