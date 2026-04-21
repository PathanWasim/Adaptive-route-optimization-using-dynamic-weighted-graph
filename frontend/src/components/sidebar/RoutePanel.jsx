import React from 'react'
import useStore from '../../store/useStore'
import { useNetwork } from '../../hooks/useNetwork'

const ALGORITHMS = [
  { value: 'dijkstra',       label: 'Dijkstra',               tag: 'O(E log V)' },
  { value: 'astar',          label: 'A* Haversine',           tag: 'O(E log V)*' },
  { value: 'bidirectional',  label: 'Bidirectional Dijkstra', tag: 'O(b^d/2)' },
  { value: 'bellman_ford',   label: 'Bellman-Ford',           tag: 'O(VE)' },
  { value: 'yen_k_shortest', label: "Yen's k-Shortest",       tag: 'O(kV(E+VlogV))' },
]

function RangeSlider({ label, value, min, max, step, onChange, cls = '' }) {
  return (
    <div className="range-wrap">
      <div className="range-header">
        <span className="range-label">{label}</span>
        <span className="range-val">{value}</span>
      </div>
      <input type="range" className={cls} min={min} max={max} step={step}
        value={value} onChange={e => onChange(parseFloat(e.target.value))} />
    </div>
  )
}

export default function RoutePanel() {
  const sourceId = useStore(s => s.sourceId)
  const targetId = useStore(s => s.targetId)
  const algorithm = useStore(s => s.algorithm)
  const kPaths = useStore(s => s.kPaths)
  const animateAlgo = useStore(s => s.animateAlgo)
  const alpha = useStore(s => s.alpha)
  const beta = useStore(s => s.beta)
  const gamma = useStore(s => s.gamma)
  const disasterType = useStore(s => s.disasterType)
  const disasterRadius = useStore(s => s.disasterRadius)
  const disasterSeverity = useStore(s => s.disasterSeverity)
  const epicenter = useStore(s => s.epicenter)
  const blockedRoads = useStore(s => s.blockedRoads)
  const computingRoute = useStore(s => s.computingRoute)
  const currentCity = useStore(s => s.currentCity)
  const setActiveTab = useStore(s => s.setActiveTab)
  const resetSelections = useStore(s => s.resetSelections)
  const { computeRoute } = useNetwork()
  const store = useStore.getState()

  const canCompute = sourceId !== null && targetId !== null && currentCity

  const handleCompute = async () => {
    const disaster = disasterType !== 'none' && epicenter
      ? { type: disasterType, epicenter, radius: disasterRadius, severity: disasterSeverity }
      : { type: 'none' }

    const data = await computeRoute({
      city_key: currentCity,
      source_id: sourceId,
      target_id: targetId,
      disaster,
      weights: { alpha, beta, gamma },
      animated: animateAlgo,
      algorithm,
      k_paths: kPaths,
      compare_algorithms: true,
      blocked_roads: blockedRoads.map(r => ({ source: r.source, target: r.target })),
    })
    if (data) setActiveTab('results')
  }

  return (
    <>
      <div className="section-title">Route Configuration</div>

      {/* Source / Target */}
      <div className="sel-grid">
        <div className={`sel-card ${sourceId !== null ? 'set-src' : ''}`}>
          <div className="sel-type">Source</div>
          <div className="sel-val">{sourceId !== null ? `#${sourceId}` : 'click map'}</div>
        </div>
        <div className={`sel-card ${targetId !== null ? 'set-tgt' : ''}`}>
          <div className="sel-type">Target</div>
          <div className="sel-val">{targetId !== null ? `#${targetId}` : 'click map'}</div>
        </div>
      </div>

      {/* Algorithm chips */}
      <div>
        <label className="field-label">Algorithm</label>
        <div className="algo-grid">
          {ALGORITHMS.map(a => (
            <div key={a.value} className={`algo-chip ${algorithm === a.value ? 'active' : ''}`}
              onClick={() => useStore.getState().setAlgorithm(a.value)}>
              <span className="algo-name">{a.label}</span>
              <span className="algo-tag">{a.tag}</span>
            </div>
          ))}
        </div>
      </div>

      {algorithm === 'yen_k_shortest' && (
        <RangeSlider label="k Paths" value={kPaths} min={1} max={5} step={1}
          onChange={v => useStore.getState().setKPaths(v)} />
      )}

      {/* Objectives */}
      <div>
        <label className="field-label">Routing Objectives</label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <RangeSlider label="Distance Weight (α)" value={alpha} min={0} max={5} step={0.1} onChange={v => useStore.getState().setAlpha(v)} />
          <RangeSlider label="Risk Avoidance (β)" value={beta}  min={0} max={5} step={0.1} onChange={v => useStore.getState().setBeta(v)} cls="warning" />
          <RangeSlider label="Congestion (γ)"     value={gamma} min={0} max={5} step={0.1} onChange={v => useStore.getState().setGamma(v)} cls="success" />
        </div>
      </div>

      {/* Toggle animate */}
      <div className="toggle-row">
        <span className="toggle-label">Show Algorithm Animation</span>
        <button className={`toggle-btn ${animateAlgo ? 'on' : ''}`}
          onClick={() => useStore.getState().setAnimateAlgo(!animateAlgo)} />
      </div>

      {/* Buttons */}
      <button className="btn btn-primary btn-full" onClick={handleCompute}
        disabled={!canCompute || computingRoute}>
        {computingRoute ? '⏳ Computing…' : '🚀 Compute & Visualize Routes'}
      </button>

      <button className="btn btn-ghost btn-full btn-sm" onClick={resetSelections}>
        ↺ Reset All Selections
      </button>

      {!canCompute && (
        <div className="text-muted" style={{ textAlign: 'center' }}>
          Select source & target nodes on the map first
        </div>
      )}
    </>
  )
}
