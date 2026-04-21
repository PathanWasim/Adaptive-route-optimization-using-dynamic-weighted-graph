import React, { useEffect } from 'react'
import Sidebar from './components/sidebar/Sidebar'
import MapView from './components/map/MapView'
import ToastSystem from './components/panels/ToastSystem'
import TopBar from './components/panels/TopBar'
import AlgoPanel from './components/panels/AlgoPanel'
import HelpModal from './components/panels/HelpModal'
import { useSocket } from './hooks/useSocket'
import useStore from './store/useStore'

export default function App() {
  useSocket() // boot WebSocket connection once

  // Global keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return
      const store = useStore.getState()
      switch (e.key.toLowerCase()) {
        case '1': store.setActiveTab('city');      break
        case '2': store.setActiveTab('route');     break
        case '3': store.setActiveTab('disaster');  break
        case '4': store.setActiveTab('results');   break
        case '5': store.setActiveTab('demo');      break
        case '6': store.setActiveTab('benchmark'); break
        case 's': store.toggleSidebar();           break
        case 'r': store.resetSelections();         break
        case 'escape': store.setBlockMode(false);  break
        case '?':
          store.setShowHelpModal(true)
          break
        default: break
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return (
    <div className="app-shell">
      <TopBar />
      <div className="app-body">
        <Sidebar />
        <MapView />
      </div>
      <AlgoPanel />
      <ToastSystem />
      <HelpModal />
    </div>
  )
}
