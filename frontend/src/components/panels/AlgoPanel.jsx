import React from 'react'
import useStore from '../../store/useStore'

export default function AlgoPanel() {
  const algoSteps = useStore(s => s.algoSteps)
  const currentStep = useStore(s => s.currentStep)
  const isAnimating = useStore(s => s.isAnimating)
  const routeData = useStore(s => s.routeData)

  if (!routeData && algoSteps.length === 0) return null

  const visitSteps = algoSteps.filter(s => s.type === 'visit')
  const pct = visitSteps.length > 0 ? Math.min(100, (currentStep / visitSteps.length) * 100) : 0
  const lastStep = visitSteps[currentStep - 1]

  return (
    <div className="algo-panel">
      <div className="algo-panel-title">
        ⚙️ Algorithm Execution
        {isAnimating && <span className="running-badge">running</span>}
      </div>

      <div className="progress-track">
        <div className="progress-fill indigo" style={{ width: `${pct}%` }} />
      </div>

      <div className="algo-stat-grid">
        {[
          { label: 'Visited', val: routeData?.normal_route?.nodes_visited ?? 0 },
          { label: 'Edges', val: routeData?.normal_route?.path_edges ?? '—' },
          { label: 'ms', val: routeData?.normal_route?.computation_time ? (routeData.normal_route.computation_time * 1000).toFixed(1) : '—' },
        ].map(({ label, val }) => (
          <div key={label} className="algo-stat">
            <div className="algo-stat-val">{val}</div>
            <div className="algo-stat-key">{label}</div>
          </div>
        ))}
      </div>

      {lastStep && (
        <div className="algo-step-box">
          Visiting Node {lastStep.node} — dist {lastStep.distance?.toFixed(1)}m
        </div>
      )}
    </div>
  )
}
