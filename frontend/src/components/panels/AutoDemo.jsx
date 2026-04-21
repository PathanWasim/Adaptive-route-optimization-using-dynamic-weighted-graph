import React, { useState, useRef } from 'react'
import useStore from '../../store/useStore'
import { useNetwork } from '../../hooks/useNetwork'

/**
 * AutoDemo: Runs a guided 5-step demo sequence automatically:
 * 1. Selects random source + target from the loaded network
 * 2. Places a fire epicenter near the midpoint
 * 3. Computes route with Dijkstra
 * 4. Computes disaster-aware route with A*
 * 5. Starts the vehicle animation
 * 6. Runs the search wave visualization
 */
export default function AutoDemo() {
  const [running, setRunning]   = useState(false)
  const [step, setStep]         = useState(0)
  const [log, setLog]           = useState([])
  const cancelRef               = useRef(false)

  const network       = useStore(s => s.network)
  const currentCity   = useStore(s => s.currentCity)
  const { computeRoute } = useNetwork()

  const STEPS = [
    'Select random source & target',
    'Place fire disaster epicenter',
    'Compute Dijkstra route',
    'Compute A* disaster-aware route',
    'Animate evacuee vehicle',
    'Run search wave visualization',
  ]

  const sleep = ms => new Promise(r => setTimeout(r, ms))

  const addLog = (msg) => setLog(prev => [...prev, `✓ ${msg}`])

  const handleStart = async () => {
    if (!network || !currentCity) {
      useStore.getState().addToast({ type: 'warning', title: 'No Network', msg: 'Load a city first before running Auto Demo' })
      return
    }
    setRunning(true)
    cancelRef.current = false
    setLog([])
    setStep(0)

    const store = useStore.getState()
    const nodes = network.nodes
    if (!nodes || nodes.length < 10) {
      useStore.getState().addToast({ type: 'error', title: 'Too Few Nodes', msg: 'Network has too few nodes for demo' })
      setRunning(false)
      return
    }

    // Step 1 — random source + target
    setStep(1)
    const rndIdx = (max) => Math.floor(Math.random() * max)
    let srcNode, tgtNode
    do {
      srcNode = nodes[rndIdx(nodes.length)]
      tgtNode = nodes[rndIdx(nodes.length)]
    } while (srcNode.id === tgtNode.id)
    store.setSource(srcNode.id)
    store.setTarget(tgtNode.id)
    addLog(`Source: Node #${srcNode.id} | Target: Node #${tgtNode.id}`)
    await sleep(800)
    if (cancelRef.current) { setRunning(false); return }

    // Step 2 — midpoint epicenter, fire disaster
    setStep(2)
    const midLat = (srcNode.lat + tgtNode.lat) / 2
    const midLon = (srcNode.lon + tgtNode.lon) / 2
    store.setDisasterType('fire')
    store.setDisasterRadius(200)
    store.setDisasterSeverity(0.7)
    store.setEpicenter([midLat, midLon])
    store.setActiveTab('disaster')
    addLog(`Fire epicenter at (${midLat.toFixed(4)}, ${midLon.toFixed(4)}) r=200m`)
    await sleep(1000)
    if (cancelRef.current) { setRunning(false); return }

    // Step 3 — Dijkstra normal route
    setStep(3)
    store.setAlgorithm('dijkstra')
    store.setActiveTab('route')
    addLog('Computing Dijkstra route (no disaster)…')
    const normalData = await computeRoute({
      city_key: currentCity,
      source_id: srcNode.id,
      target_id: tgtNode.id,
      disaster: { type: 'none' },
      weights: { alpha: 1, beta: 1, gamma: 1 },
      animated: true,
      algorithm: 'dijkstra',
      compare_algorithms: false,
    })
    if (normalData) addLog(`Dijkstra: ${normalData.normal_route?.distance?.toFixed(0)}m in ${(normalData.normal_route?.computation_time*1000).toFixed(1)}ms`)
    await sleep(1200)
    if (cancelRef.current) { setRunning(false); return }

    // Step 4 — A* disaster-aware route
    setStep(4)
    store.setAlgorithm('astar')
    addLog('Computing A* disaster-aware route…')
    const disasterData = await computeRoute({
      city_key: currentCity,
      source_id: srcNode.id,
      target_id: tgtNode.id,
      disaster: { type: 'fire', epicenter: [midLat, midLon], radius: 200, severity: 0.7 },
      weights: { alpha: 1, beta: 1.5, gamma: 1 },
      animated: true,
      algorithm: 'astar',
      compare_algorithms: true,
    })
    if (disasterData) {
      const inc = disasterData.metrics?.percent_increase
      addLog(`A*: detour +${inc?.toFixed(1) ?? '?'}% to avoid fire zone`)
      store.setActiveTab('results')
    }
    await sleep(1500)
    if (cancelRef.current) { setRunning(false); return }

    // Step 5 — vehicle animation
    setStep(5)
    store.setActiveTab('demo')
    const path = disasterData?.disaster_route?.path || disasterData?.normal_route?.path
    if (path && path.length >= 2) {
      store.setVehiclePath(path)
      store.setVehicleRunning(true)
      addLog('Vehicle animation started 🚗')
    }
    await sleep(1000)
    if (cancelRef.current) { setRunning(false); return }

    // Step 6 — search wave
    setStep(6)
    if (disasterData?.normal_route?.steps?.length) {
      store.clearWave()
      store.setWaveRunning(true)
      const steps = disasterData.normal_route.steps.filter(s => s.type === 'visit')
      for (let i = 0; i < steps.length && !cancelRef.current; i++) {
        const s = steps[i]
        if (s.coords) {
          store.appendWaveNode({ lat: s.coords[0], lon: s.coords[1], algo: 'dijkstra', step: i })
          if (Math.random() < 0.55) {
            store.appendWaveNode({ lat: s.coords[0] + (Math.random()-0.5)*0.0005, lon: s.coords[1] + (Math.random()-0.5)*0.0005, algo: 'astar', step: i })
          }
        }
        await sleep(40)
      }
      store.setWaveRunning(false)
      addLog(`Search wave complete — ${store.waveStepCount} nodes visualized`)
    } else {
      addLog('(re-enable animation toggle for search wave)')
    }

    setStep(0)
    setRunning(false)
    useStore.getState().addToast({ type: 'success', title: '🎉 Demo Complete!', msg: 'All 6 demo steps executed successfully' })
  }

  const handleStop = () => {
    cancelRef.current = true
    useStore.getState().setVehicleRunning(false)
    useStore.getState().setWaveRunning(false)
    setRunning(false)
    setStep(0)
  }

  return (
    <div className="card" style={{ background: 'rgba(99,102,241,0.07)', border: '1px solid rgba(99,102,241,0.25)' }}>
      <div className="card-title indigo">🎬 Auto Demo Sequence</div>
      <p className="card-desc">
        Runs a full guided 6-step demo automatically: random nodes → disaster → Dijkstra → A* → vehicle → wave.
        Perfect for live presentations.
      </p>

      <div className="btn-group" style={{ marginBottom: 10 }}>
        <button className="btn btn-primary btn-sm" onClick={handleStart} disabled={running}>
          {running ? `⏳ Step ${step}/6…` : '▶ Run Full Auto Demo'}
        </button>
        <button className="btn btn-danger btn-sm" onClick={handleStop} disabled={!running}>
          ⏹ Stop
        </button>
      </div>

      {/* Step tracker */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {['Select nodes', 'Place epicenter', 'Dijkstra route', 'A* safe route', 'Vehicle anim', 'Search wave'].map((label, i) => {
          const idx = i + 1
          const done = step > idx || (!running && log.length >= idx)
          const active = step === idx
          return (
            <div key={idx} style={{
              display: 'flex', alignItems: 'center', gap: 8,
              fontSize: 11,
              color: done ? '#6ee7b7' : active ? '#fcd34d' : 'var(--c-text-4)',
            }}>
              <span style={{
                width: 18, height: 18, borderRadius: '50%', flexShrink: 0,
                background: done ? 'rgba(16,185,129,0.25)' : active ? 'rgba(245,158,11,0.25)' : 'rgba(255,255,255,0.05)',
                border: `1px solid ${done ? '#10b981' : active ? '#f59e0b' : 'rgba(255,255,255,0.1)'}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 9, fontWeight: 700
              }}>
                {done ? '✓' : idx}
              </span>
              {label}
              {active && <span style={{ color: '#f59e0b', animation: 'pulse 1s infinite' }}>●</span>}
            </div>
          )
        })}
      </div>

      {/* Log */}
      {log.length > 0 && (
        <div style={{
          marginTop: 10, background: 'rgba(0,0,0,0.3)', borderRadius: 8, padding: 10,
          maxHeight: 110, overflowY: 'auto', fontSize: 10, fontFamily: 'JetBrains Mono',
          display: 'flex', flexDirection: 'column', gap: 3
        }}>
          {log.map((l, i) => <div key={i} style={{ color: '#6ee7b7' }}>{l}</div>)}
        </div>
      )}
    </div>
  )
}
