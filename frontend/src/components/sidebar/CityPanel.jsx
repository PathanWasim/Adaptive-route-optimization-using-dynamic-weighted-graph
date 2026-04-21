import React, { useEffect, useState } from 'react'
import useStore from '../../store/useStore'
import { useNetwork } from '../../hooks/useNetwork'

export default function CityPanel() {
  const [cities, setCities] = useState([])
  const [selected, setSelected] = useState('')
  const loadingNetwork = useStore(s => s.loadingNetwork)
  const network = useStore(s => s.network)
  const setActiveTab = useStore(s => s.setActiveTab)
  const { fetchCities, loadNetwork } = useNetwork()

  useEffect(() => { fetchCities().then(setCities).catch(console.error) }, [])

  const handleLoad = async () => {
    const data = await loadNetwork(selected)
    if (data) setActiveTab('route')
  }

  return (
    <>
      <div className="section-title">Select City</div>

      <div>
        <label className="field-label">Road Network</label>
        <select className="input-select" value={selected} onChange={e => setSelected(e.target.value)}>
          <option value="">— Select a city —</option>
          {cities.map(c => <option key={c.key} value={c.key}>{c.name}</option>)}
        </select>
      </div>

      <button
        className="btn btn-primary btn-full"
        onClick={handleLoad}
        disabled={!selected || loadingNetwork}
      >
        {loadingNetwork ? '⏳ Loading Network…' : '🌐 Load Road Network'}
      </button>

      {network && (
        <div className="info-box success">
          <b>✓ Network Loaded</b><br />
          {network.stats?.num_nodes?.toLocaleString()} nodes · {network.stats?.num_edges?.toLocaleString()} edges
        </div>
      )}

      <div className="info-box info">
        <b>ℹ Academic Integrity</b><br />
        ✓ All routing uses internal algorithms (Dijkstra, A*, etc.)<br />
        ✓ Maps used for visualization only<br />
        ✓ No external routing APIs
      </div>
    </>
  )
}
