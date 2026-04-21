import React from 'react'
import useStore from '../../store/useStore'

export default function HelpModal() {
  const showHelpModal = useStore(s => s.showHelpModal)
  const setShowHelpModal = useStore(s => s.setShowHelpModal)
  const setActiveTab = useStore(s => s.setActiveTab)

  if (!showHelpModal) return null

  return (
    <div className="modal-overlay" onClick={() => setShowHelpModal(false)}>
      <div className="modal-content card" onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h2 style={{ fontSize: 18, fontWeight: 800, margin: 0, color: 'var(--c-text)' }}>
            🚨 Welcome to EvacRoute
          </h2>
          <button className="toast-close" style={{ fontSize: 20 }} onClick={() => setShowHelpModal(false)}>✕</button>
        </div>

        <p style={{ fontSize: 12, color: 'var(--c-text-2)', lineHeight: 1.5, marginBottom: 20 }}>
          EvacRoute is an interactive, real-time disaster evacuation routing system.
          It visualizes how algorithms adapt to dynamic hazards like fires and floods.
        </p>

        <div className="metric-grid" style={{ gridTemplateColumns: '1fr', gap: 12, marginBottom: 20 }}>
          <div className="metric-card" style={{ borderLeftColor: 'var(--c-primary)' }}>
            <div className="metric-label" style={{ color: 'var(--c-primary)' }}>Option A: Guided Tour</div>
            <div style={{ fontSize: 13, color: 'var(--c-text)', fontWeight: 600, marginTop: 4 }}>
              The easiest way to see what EvacRoute can do.
            </div>
            <ul style={{ paddingLeft: 16, marginTop: 8, fontSize: 11, color: 'var(--c-text-3)', lineHeight: 1.6 }}>
              <li>Go to the <b>City</b> tab and load a network.</li>
              <li>Go to the <b>Demo</b> tab (🎮).</li>
              <li>Click <b>▶ Run Full Auto Demo</b> to watch a guided 6-step presentation.</li>
            </ul>
            <button className="btn btn-primary btn-sm" style={{ marginTop: 10 }} onClick={() => { setShowHelpModal(false); setActiveTab('demo') }}>
              Take me to the Demo tab
            </button>
          </div>

          <div className="metric-card" style={{ borderLeftColor: 'var(--c-warning)' }}>
            <div className="metric-label" style={{ color: 'var(--c-warning)' }}>Option B: Explore Manually</div>
            <div style={{ fontSize: 13, color: 'var(--c-text)', fontWeight: 600, marginTop: 4 }}>
              Build your own scenarios and break things.
            </div>
            <ul style={{ paddingLeft: 16, marginTop: 8, fontSize: 11, color: 'var(--c-text-3)', lineHeight: 1.6 }}>
              <li><b>1/ City:</b> Load a road network map.</li>
              <li><b>2/ Route / Map:</b> Click on the map to set Source 🟢 and Target 🔴.</li>
              <li><b>3/ Hazard:</b> Pick a disaster type and place the epicenter on the map.</li>
              <li><b>4/ Route:</b> Hit compute! Then check <b>Results</b> or use <b>Block Mode</b>.</li>
            </ul>
          </div>
        </div>

        <div className="info-box info">
          <b>⌨ Keyboard Shortcuts:</b> Use numbers <b>1-6</b> to switch between tabs, <b>s</b> to toggle sidebar, <b>r</b> to reset the map.
        </div>
      </div>
    </div>
  )
}
