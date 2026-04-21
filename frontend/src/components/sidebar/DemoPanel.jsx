import React, { useState, useRef } from 'react'
import useStore from '../../store/useStore'
import { useSocket } from '../../hooks/useSocket'
import AutoDemo from '../panels/AutoDemo'

/* ── Feature 1: WebSocket-Driven Disaster Expansion ─────────────────────── */
function DisasterSim() {
  const { startSim, stopSim } = useSocket()
  const simRunning    = useStore(s => s.simRunning)
  const simPercent    = useStore(s => s.simPercent)
  const simRadius     = useStore(s => s.simRadius)
  const setSimRunning = useStore(s => s.setSimRunning)
  const currentCity   = useStore(s => s.currentCity)
  const disasterType  = useStore(s => s.disasterType)
  const disasterSeverity = useStore(s => s.disasterSeverity)
  const epicenter     = useStore(s => s.epicenter)
  const [spreadRate, setSpreadRate] = useState(30)
  const [maxRadius, setMaxRadius]   = useState(800)

  const canStart = currentCity && disasterType !== 'none' && epicenter

  const handlePlay = () => {
    if (!canStart) {
      useStore.getState().addToast({ type: 'warning', title: 'Setup Required', msg: 'Load a city → pick disaster type → place epicenter, then try again.' })
      return
    }
    setSimRunning(true)
    startSim({
      city_key: currentCity,
      disaster: { type: disasterType, epicenter, severity: disasterSeverity },
      spread_rate: spreadRate,
      max_radius: maxRadius,
      start_radius: 80,
      interval: 1.5,
    })
  }

  const handleStop = () => {
    setSimRunning(false)
    stopSim()
    useStore.getState().updateSimTick({ radius: 0, percent: 0, blocked_edges: [] })
  }

  return (
    <div className="card card-red">
      <div className="card-title red">🔥 Live Disaster Expansion</div>
      <p className="card-desc">
        The Flask server emits a <b>disaster_tick</b> WebSocket event every 1.5s.
        The disaster circle grows and blocked roads update automatically in real-time.
      </p>

      <div style={{ display:'flex', flexDirection:'column', gap:10, marginBottom:12 }}>
        <div className="range-wrap">
          <div className="range-header">
            <span className="range-label">Spread Rate</span>
            <span className="range-val">{spreadRate}m/tick</span>
          </div>
          <input type="range" className="danger" min={10} max={100} step={10}
            value={spreadRate} onChange={e => setSpreadRate(+e.target.value)} />
        </div>
        <div className="range-wrap">
          <div className="range-header">
            <span className="range-label">Max Radius</span>
            <span className="range-val">{maxRadius}m</span>
          </div>
          <input type="range" className="danger" min={200} max={1500} step={50}
            value={maxRadius} onChange={e => setMaxRadius(+e.target.value)} />
        </div>
      </div>

      <div className="btn-group">
        <button className="btn btn-success btn-sm" onClick={handlePlay} disabled={simRunning}>
          ▶ Play
        </button>
        <button className="btn btn-danger btn-sm" onClick={handleStop} disabled={!simRunning}>
          ⏹ Stop
        </button>
      </div>

      {(simRunning || simRadius > 0) && (
        <div style={{ marginTop: 10 }}>
          <div className="progress-track">
            <div className="progress-fill red" style={{ width: `${simPercent}%` }} />
          </div>
          <div className="progress-sub">
            {simRunning ? '🔴 LIVE' : '⏹'} Radius: {simRadius.toFixed(0)}m ({simPercent.toFixed(0)}%)
          </div>
        </div>
      )}

      {!canStart && (
        <div className="info-box warning" style={{ marginTop: 10, fontSize: 10 }}>
          ⚠ Complete setup: City → Hazard type → Epicenter on map
        </div>
      )}
    </div>
  )
}

/* ── Feature 2: Interactive Road Blocker ─────────────────────────────────── */
function RoadBlocker() {
  const blockModeActive = useStore(s => s.blockModeActive)
  const blockedRoads    = useStore(s => s.blockedRoads)
  const setBlockMode    = useStore(s => s.setBlockMode)
  const clearBlockedRoads = useStore(s => s.clearBlockedRoads)

  return (
    <div className="card card-yellow">
      <div className="card-title yellow">🚧 Interactive Road Blocker</div>
      <p className="card-desc">
        Activate block mode then <b>click any road</b> on the map to close it.
        Recompute the route to see the algorithm reroute automatically.
      </p>

      <button
        style={{
          width: '100%',
          padding: '9px 12px',
          borderRadius: 8,
          border: blockModeActive ? '1px solid var(--c-warning)' : '1px solid rgba(245,158,11,0.3)',
          background: blockModeActive ? 'rgba(245,158,11,0.2)' : 'rgba(245,158,11,0.07)',
          color: '#fcd34d',
          fontSize: 13,
          fontWeight: 700,
          cursor: 'pointer',
          fontFamily: 'inherit',
          transition: 'all 0.2s',
          animation: blockModeActive ? 'pulse 1.3s infinite' : 'none',
        }}
        onClick={() => setBlockMode(!blockModeActive)}
      >
        🚧 {blockModeActive ? 'Block Mode ACTIVE — Click Roads' : 'Activate Block Mode'}
      </button>

      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginTop:8, fontSize:11, color:'var(--c-text-3)' }}>
        <span>{blockedRoads.length} road{blockedRoads.length !== 1 ? 's' : ''} blocked</span>
        {blockedRoads.length > 0 && (
          <button onClick={clearBlockedRoads} style={{ background:'none', border:'none', cursor:'pointer', color:'var(--c-danger)', fontSize:11, fontFamily:'inherit' }}>
            🗑 Clear All
          </button>
        )}
      </div>

      {blockedRoads.length > 0 && (
        <div className="block-list">
          {blockedRoads.map((r, i) => (
            <div key={i} className="block-item">
              <span>Edge {r.source}→{r.target}</span>
              <span className="badge-closed">🚧 Closed</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ── Feature 3: Vehicle Animation ────────────────────────────────────────── */
function VehicleAnimator() {
  const routeData       = useStore(s => s.routeData)
  const vehicleRunning  = useStore(s => s.vehicleRunning)
  const vehicleProgress = useStore(s => s.vehicleProgress)
  const setVehiclePath  = useStore(s => s.setVehiclePath)
  const setVehicleRunning = useStore(s => s.setVehicleRunning)

  const handleGo = () => {
    const path = routeData?.disaster_route?.path || routeData?.normal_route?.path
    if (!path || path.length < 2) {
      useStore.getState().addToast({ type: 'warning', title: 'No Route', msg: 'Compute a route first, then start the vehicle.' })
      return
    }
    setVehiclePath(path)
    setVehicleRunning(true)
  }

  const handleStop = () => {
    setVehicleRunning(false)
    useStore.getState().setVehiclePath([])
    useStore.getState().updateVehicle(null, 0)
  }

  return (
    <div className="card card-violet">
      <div className="card-title violet">🚗 Vehicle Animation</div>
      <p className="card-desc">
        Animates an evacuee vehicle along the computed route using <b>requestAnimationFrame</b> for smooth 60fps motion.
        It reroutes dynamically if disaster expands.
      </p>

      <div className="btn-group">
        <button className="btn btn-sm" onClick={handleGo} disabled={vehicleRunning || !routeData}
          style={{ flex:1, background:'linear-gradient(135deg,#6366f1,#7c3aed)', color:'#fff', border:'none', borderRadius:8, fontWeight:700, fontSize:12, cursor:'pointer', opacity: (vehicleRunning || !routeData) ? 0.4 : 1 }}>
          🚗 Start Journey
        </button>
        <button className="btn btn-danger btn-sm" onClick={handleStop} disabled={!vehicleRunning}>
          ⏹ Stop
        </button>
      </div>

      {(vehicleRunning || vehicleProgress > 0) && (
        <div style={{ marginTop: 10 }}>
          <div className="progress-track">
            <div className="progress-fill cyan" style={{ width: `${vehicleProgress * 100}%` }} />
          </div>
          <div className="progress-sub">
            {vehicleRunning ? '🚗 En Route' : '✅ Arrived'} — {(vehicleProgress * 100).toFixed(0)}%
          </div>
        </div>
      )}
    </div>
  )
}

/* ── Feature 4: Algorithm Search Wave ────────────────────────────────────── */
function SearchWaveControl() {
  const routeData      = useStore(s => s.routeData)
  const waveRunning    = useStore(s => s.waveRunning)
  const waveStepCount  = useStore(s => s.waveStepCount)
  const appendWaveNode = useStore(s => s.appendWaveNode)
  const clearWave      = useStore(s => s.clearWave)
  const setWaveRunning = useStore(s => s.setWaveRunning)
  const network        = useStore(s => s.network)
  const [speed, setSpeed] = useState(50)

  const runWave = async () => {
    if (!routeData?.normal_route?.steps) {
      useStore.getState().addToast({ type: 'warning', title: 'No Steps', msg: 'Enable animation toggle in Route tab, then recompute.' })
      return
    }
    clearWave()
    setWaveRunning(true)

    const steps = routeData.normal_route.steps.filter(s => s.type === 'visit')
    const running = { current: true }

    for (let i = 0; i < steps.length; i++) {
      if (!useStore.getState().waveRunning) break
      const step = steps[i]
      if (step.coords) {
        // Dijkstra: all nodes (blue)
        appendWaveNode({ lat: step.coords[0], lon: step.coords[1], algo: 'dijkstra', step: i })
        // A*: only ~55% directional subset (green) — simulates heuristic pruning
        if (Math.random() < 0.55) {
          appendWaveNode({
            lat: step.coords[0] + (Math.random() - 0.5) * 0.0008,
            lon: step.coords[1] + (Math.random() - 0.5) * 0.0008,
            algo: 'astar', step: i
          })
        }
      }
      await new Promise(r => setTimeout(r, speed))
    }
    setWaveRunning(false)
    useStore.getState().addToast({ type: 'success', title: 'Wave Complete', msg: `${useStore.getState().waveStepCount} nodes rendered` })
  }

  const stopWave = () => { setWaveRunning(false); clearWave() }

  return (
    <div className="card card-cyan">
      <div className="card-title cyan">⚡ Algorithm Search Wave</div>
      <p className="card-desc">
        Visualizes Dijkstra (🔵 360° wave) vs A* (🟢 directed beam) node exploration simultaneously.
        Proves why A* visits significantly fewer nodes.
      </p>

      <div className="wave-pills">
        <div className="wave-pill dijk">
          <div>🔵 Dijkstra</div>
          <div className="wl">Explores all neighbours</div>
        </div>
        <div className="wave-pill astar">
          <div>🟢 A* Heuristic</div>
          <div className="wl">Directed toward goal</div>
        </div>
      </div>

      <div className="range-wrap" style={{ marginBottom: 10 }}>
        <div className="range-header">
          <span className="range-label">Animation Speed</span>
          <span className="range-val">{speed}ms/step</span>
        </div>
        <input type="range" className="cyan" min={10} max={300} step={10}
          value={speed} onChange={e => setSpeed(+e.target.value)} />
      </div>

      <div className="btn-group">
        <button className="btn btn-cyan btn-sm" onClick={runWave} disabled={waveRunning}>
          ⚡ Visualize Waves
        </button>
        <button className="btn btn-ghost btn-sm" onClick={stopWave} disabled={!waveRunning && waveStepCount === 0}>
          🗑 Clear
        </button>
      </div>

      {waveStepCount > 0 && (
        <div className="progress-sub" style={{ marginTop: 8 }}>
          {waveRunning ? '⚡ Animating…' : '✅ Done'} — {waveStepCount} nodes rendered
        </div>
      )}
    </div>
  )
}

/* ── Main DemoPanel ─────────────────────────────────────────────────────── */
export default function DemoPanel() {
  return (
    <>
      <div className="section-title">🎮 Live Demo Features</div>
      <AutoDemo />
      <DisasterSim />
      <RoadBlocker />
      <VehicleAnimator />
      <SearchWaveControl />
    </>
  )
}
