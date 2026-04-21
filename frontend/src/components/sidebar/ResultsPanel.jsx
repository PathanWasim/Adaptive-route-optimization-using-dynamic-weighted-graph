import React from 'react'
import useStore from '../../store/useStore'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts'

const ALGO_MAP = {
  dijkstra: { name: 'Dijkstra', tag: 'O(E log V)', color: '#6366f1' },
  astar: { name: 'A*', tag: 'O(E log V)*', color: '#10b981' },
  bidirectional: { name: 'Bi-Dijkstra', tag: 'O(b^d/2)', color: '#06b6d4' },
  bellman_ford: { name: 'Bellman-Ford', tag: 'O(VE)', color: '#f59e0b' },
  yen_k_shortest: { name: "Yen k-SP", tag: 'O(kV)', color: '#ec4899' },
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: '#0d1426', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, padding: '8px 12px', fontSize: 11 }}>
      <b style={{ color: '#fff' }}>{label}</b>
      <div style={{ color: '#94a3b8', marginTop: 3 }}>{payload[0].value.toFixed(2)} ms</div>
    </div>
  )
}

export default function ResultsPanel() {
  const routeData    = useStore(s => s.routeData)
  const routeHistory = useStore(s => s.routeHistory)

  if (!routeData) {
    return (
      <div className="empty-state">
        <div className="em-icon">📊</div>
        No routes computed yet.<br />
        <span className="text-muted">Set source &amp; target, then compute a route.</span>
      </div>
    )
  }

  const { normal_route, disaster_route, metrics, algorithm_comparison } = routeData

  const chartData = algorithm_comparison
    ? Object.entries(algorithm_comparison).map(([key, d]) => ({
        name: ALGO_MAP[key]?.name || key,
        time: parseFloat(d.time_ms.toFixed(2)),
        nodes: d.nodes_visited,
        color: ALGO_MAP[key]?.color || '#6366f1',
        key,
      }))
    : []

  const minTime = chartData.length ? Math.min(...chartData.map(d => d.time)) : 0

  return (
    <>
      <div className="section-title">Route Results</div>

      {/* Metric grid */}
      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-label">Normal Route</div>
          <div className="metric-value">{normal_route?.distance?.toFixed(0)}<span style={{fontSize:11,fontWeight:400,fontFamily:'Inter'}}>m</span></div>
          <div className="metric-sub">{(normal_route?.computation_time * 1000).toFixed(1)}ms compute</div>
        </div>
        {disaster_route && (
          <div className="metric-card green">
            <div className="metric-label">Safe Route</div>
            <div className="metric-value">{disaster_route?.distance?.toFixed(0)}<span style={{fontSize:11,fontWeight:400,fontFamily:'Inter'}}>m</span></div>
            <div className="metric-sub">Disaster-aware</div>
          </div>
        )}
        <div className="metric-card cyan">
          <div className="metric-label">Nodes Visited</div>
          <div className="metric-value">{normal_route?.nodes_visited}</div>
        </div>
        {metrics?.percent_increase !== undefined && (
          <div className="metric-card red">
            <div className="metric-label">Detour</div>
            <div className="metric-value">+{metrics.percent_increase?.toFixed(1)}%</div>
            <div className="metric-sub">{metrics.routes_diverged ? 'Route diverged ✓' : ''}</div>
          </div>
        )}
      </div>

      {/* Bar Chart */}
      {chartData.length > 0 && (
        <>
          <div className="field-label">Algorithm Speed Comparison</div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={120}>
              <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -22 }}>
                <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="time" radius={[4,4,0,0]}>
                  {chartData.map(d => (
                    <Cell key={d.key} fill={d.time === minTime ? '#10b981' : d.color} fillOpacity={d.time === minTime ? 1 : 0.65} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="chart-subtitle">Time (ms) · green = fastest</div>
          </div>

          {/* Comparison table */}
          <table className="algo-table">
            <thead>
              <tr>
                <th>Algorithm</th>
                <th style={{ textAlign:'right' }}>Time</th>
                <th style={{ textAlign:'right' }}>Nodes</th>
                <th style={{ textAlign:'right' }}>Complexity</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(algorithm_comparison).map(([key, d]) => {
                const isBest = d.time_ms === minTime
                return (
                  <tr key={key} className={isBest ? 'best' : ''}>
                    <td className="name">
                      {ALGO_MAP[key]?.name || key}
                      {isBest && <span className="best-badge">fastest</span>}
                    </td>
                    <td className="time" style={{ textAlign:'right' }}>{d.time_ms.toFixed(2)}ms</td>
                    <td style={{ textAlign:'right' }}>{d.nodes_visited}</td>
                    <td style={{ textAlign:'right', fontFamily:'JetBrains Mono', fontSize:9, color:'var(--c-text-4)' }}>
                      {ALGO_MAP[key]?.tag || '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </>
      )}

      {/* History */}
      {routeHistory.length > 0 && (
        <>
          <div className="field-label" style={{ marginTop: 4 }}>Route History</div>
          {routeHistory.slice(0, 6).map((r, i) => (
            <div key={i} className="history-row">
              <span className="history-ts">{r.ts}</span>
              <span className="history-algo">{ALGO_MAP[r.algo]?.name || r.algo}</span>
              <span className="history-dist">{r.dist?.toFixed(0)}m</span>
              {r.disaster_dist && <span className="history-ddist">→{r.disaster_dist?.toFixed(0)}m</span>}
            </div>
          ))}
        </>
      )}
    </>
  )
}
