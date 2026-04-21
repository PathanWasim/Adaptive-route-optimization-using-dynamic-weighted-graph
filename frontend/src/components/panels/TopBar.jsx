import React from 'react'
import useStore from '../../store/useStore'

export default function TopBar() {
  const currentCity = useStore(s => s.currentCity)
  const tbNodes = useStore(s => s.tbNodes)
  const tbEdges = useStore(s => s.tbEdges)
  const simRunning = useStore(s => s.simRunning)
  const routeData = useStore(s => s.routeData)

  return (
    <div className="topbar">
      <div className="topbar-logo">EVAC<span>ROUTE</span></div>
      <div className={`topbar-dot ${simRunning ? 'live' : ''}`} />

      {currentCity && <>
        <div className="topbar-chip">📍 <b>{currentCity.replace(/_/g,' ')}</b></div>
        <div className="topbar-chip"><b>{tbNodes}</b> nodes</div>
        <div className="topbar-chip"><b>{tbEdges}</b> edges</div>
      </>}

      {routeData && (
        <div className="topbar-chip indigo">
          🗺 {routeData.algorithm} · {routeData.normal_route?.distance?.toFixed(0)}m
        </div>
      )}

      {simRunning && (
        <div className="topbar-chip red">⚡ LIVE DISASTER SIM</div>
      )}

      <div className="topbar-spacer" />
      <div className="topbar-ws" style={{ marginRight: 12 }}>
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M5 12.55a11 11 0 0 1 14.08 0M1.42 9a16 16 0 0 1 21.16 0M8.53 16.11a6 6 0 0 1 6.95 0M12 20h.01"/>
        </svg>
        WebSocket
      </div>
      <button className="topbar-chip" style={{ cursor: 'pointer', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--c-border2)' }} onClick={() => useStore.getState().setShowHelpModal(true)}>
        ❓ Help
      </button>
    </div>
  )
}
