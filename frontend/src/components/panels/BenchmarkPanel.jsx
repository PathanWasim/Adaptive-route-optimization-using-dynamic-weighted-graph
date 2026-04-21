import React, { useState } from 'react'
import useStore from '../../store/useStore'
import { useNetwork } from '../../hooks/useNetwork'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid
} from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#0d1426', border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 8, padding: '8px 12px', fontSize: 11
    }}>
      <b style={{ color: '#fff' }}>V = {label}</b>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.color, marginTop: 2 }}>
          {p.name}: {p.value?.toFixed(2)}ms
        </div>
      ))}
    </div>
  )
}

export default function BenchmarkPanel() {
  const [running, setRunning]     = useState(false)
  const [results, setResults]     = useState(null)
  const [error, setError]         = useState(null)
  const { runBenchmark }          = useNetwork()
  const addToast                  = useStore(s => s.addToast)

  const handleRun = async () => {
    setRunning(true)
    setError(null)
    setResults(null)
    try {
      addToast({ type: 'info', title: 'Benchmarks Running', msg: 'Testing all algorithms — this may take ~30s' })
      const data = await runBenchmark([50, 100, 200, 500, 1000])
      setResults(data)
      addToast({ type: 'success', title: 'Benchmarks Complete', msg: `${data.results?.length} graph sizes tested` })
    } catch (e) {
      setError(e.message)
      addToast({ type: 'error', title: 'Benchmark Error', msg: e.message })
    } finally {
      setRunning(false)
    }
  }

  const chartData = results?.results?.map(r => ({
    V: r.vertices,
    Dijkstra: r.dijkstra?.avg_time_ms ?? null,
    'A*': r.astar?.avg_time_ms ?? null,
    'Bi-Dijkstra': r.bidirectional?.avg_time_ms ?? null,
    'Bellman-Ford': r.bellman_ford?.avg_time_ms ?? null,
  })) ?? []

  return (
    <div className="card card-indigo" style={{ marginTop: 0 }}>
      <div className="card-title indigo">📊 Empirical Complexity Benchmarks</div>
      <p className="card-desc">
        Runs all algorithms on synthetic graphs of increasing size (V = 50 → 1000) to
        empirically demonstrate O(E log V) vs O(VE) runtime scaling.
      </p>

      <button
        className={`btn btn-full btn-sm ${running ? 'btn-ghost' : 'btn-primary'}`}
        onClick={handleRun}
        disabled={running}
      >
        {running ? '⏳ Running Benchmarks…' : '▶ Run Complexity Benchmarks'}
      </button>

      {error && (
        <div className="info-box" style={{ marginTop: 10, background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5', fontSize: 11 }}>
          ⚠ {error}
        </div>
      )}

      {results && chartData.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div className="field-label">Runtime Scaling (ms)</div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="V" tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 10, color: '#64748b' }} />
                <Line type="monotone" dataKey="Dijkstra"     stroke="#6366f1" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="A*"           stroke="#10b981" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Bi-Dijkstra"  stroke="#06b6d4" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Bellman-Ford" stroke="#f59e0b" strokeWidth={2} dot={false} strokeDasharray="4 2" />
              </LineChart>
            </ResponsiveContainer>
            <div className="chart-subtitle">Graph size V vs. avg runtime ms · 3 runs each</div>
          </div>

          {/* Table */}
          <table className="algo-table" style={{ marginTop: 10 }}>
            <thead>
              <tr>
                <th>V</th><th style={{ textAlign:'right' }}>E</th>
                <th style={{ textAlign:'right' }}>Dijkstra</th>
                <th style={{ textAlign:'right' }}>A*</th>
                <th style={{ textAlign:'right' }}>Bi-Dijk</th>
                <th style={{ textAlign:'right' }}>B-Ford</th>
              </tr>
            </thead>
            <tbody>
              {results.results.map(r => (
                <tr key={r.vertices}>
                  <td className="name">{r.vertices}</td>
                  <td style={{ textAlign:'right' }}>{r.edges}</td>
                  <td className="time" style={{ textAlign:'right' }}>{r.dijkstra?.avg_time_ms?.toFixed(2) ?? '—'}</td>
                  <td className="time" style={{ textAlign:'right' }}>{r.astar?.avg_time_ms?.toFixed(2) ?? '—'}</td>
                  <td className="time" style={{ textAlign:'right' }}>{r.bidirectional?.avg_time_ms?.toFixed(2) ?? '—'}</td>
                  <td className="time" style={{ textAlign:'right' }}>{r.bellman_ford?.avg_time_ms?.toFixed(2) ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* A* node reduction stats */}
          {results.results.some(r => r.astar?.node_reduction_pct != null) && (
            <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div className="field-label">A* Node Reduction vs Dijkstra</div>
              {results.results.filter(r => r.astar?.node_reduction_pct != null).map(r => (
                <div key={r.vertices} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, padding: '4px 0', borderBottom: '1px solid var(--c-border2)' }}>
                  <span style={{ color: 'var(--c-text-3)' }}>V = {r.vertices}</span>
                  <span style={{ color: '#6ee7b7', fontWeight: 700, fontFamily: 'JetBrains Mono' }}>
                    -{r.astar.node_reduction_pct.toFixed(1)}% nodes
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
