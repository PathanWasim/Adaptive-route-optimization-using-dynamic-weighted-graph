import React from 'react'
import useStore from '../../store/useStore'
import CityPanel from './CityPanel'
import RoutePanel from './RoutePanel'
import DisasterPanel from './DisasterPanel'
import ResultsPanel from './ResultsPanel'
import DemoPanel from './DemoPanel'
import BenchmarkPanel from '../panels/BenchmarkPanel'

const TABS = [
  { id: 'city',      icon: '🌍', label: 'City' },
  { id: 'route',     icon: '🗺️', label: 'Route' },
  { id: 'disaster',  icon: '⚠️', label: 'Hazard' },
  { id: 'results',   icon: '📊', label: 'Results' },
  { id: 'demo',      icon: '🎮', label: 'Demo' },
  { id: 'benchmark', icon: '⚡', label: 'Bench' },
]

const PANELS = {
  city: CityPanel,
  route: RoutePanel,
  disaster: DisasterPanel,
  results: ResultsPanel,
  demo: DemoPanel,
  benchmark: BenchmarkPanel,
}

export default function Sidebar() {
  const activeTab     = useStore(s => s.activeTab)
  const sidebarOpen   = useStore(s => s.sidebarOpen)
  const setActiveTab  = useStore(s => s.setActiveTab)
  const toggleSidebar = useStore(s => s.toggleSidebar)
  const network       = useStore(s => s.network)
  const routeData     = useStore(s => s.routeData)
  const ActivePanel   = PANELS[activeTab] ?? CityPanel

  const getTabClass = (id) => {
    let cls = `sb-tab ${activeTab === id ? 'active' : ''}`
    if (activeTab !== id) {
      if (!network && id === 'city') cls += ' tab-pulse'
      if (network && !routeData && id === 'route') cls += ' tab-pulse'
    }
    return cls
  }

  return (
    <div className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
      {/* Header */}
      <div className="sb-header">
        <div className="sb-logo-icon">🚨</div>
        {sidebarOpen && (
          <div className="sb-logo-text">
            <h1>EvacRoute</h1>
            <p>Adaptive Disaster Routing System</p>
          </div>
        )}
        <button className="sb-collapse-btn" onClick={toggleSidebar} title="Toggle sidebar">
          {sidebarOpen ? '◀' : '▶'}
        </button>
      </div>

      {/* Tab bar */}
      <div className="sb-tabs">
        {TABS.map(({ id, icon, label }) => (
          <button
            key={id}
            className={getTabClass(id)}
            onClick={() => { setActiveTab(id); if (!sidebarOpen) toggleSidebar() }}
            title={label}
          >
            <span style={{ fontSize: 15 }}>{icon}</span>
            {sidebarOpen && label}
          </button>
        ))}
      </div>

      {/* Active panel */}
      {sidebarOpen && (
        <div className="sb-panel">
          <div className="panel-inner">
            <ActivePanel />
          </div>
        </div>
      )}
    </div>
  )
}
