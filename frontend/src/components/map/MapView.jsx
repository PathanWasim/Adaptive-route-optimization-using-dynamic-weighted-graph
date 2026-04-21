import React, { useCallback } from 'react'
import {
  MapContainer, TileLayer, Polyline, CircleMarker, Circle, Popup, useMapEvents, useMap
} from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import useStore from '../../store/useStore'
import VehicleMarker from './VehicleMarker'
import SearchWave from './SearchWave'

delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

const DISASTER_COLORS = { fire: '#ef4444', flood: '#3b82f6', earthquake: '#f59e0b' }

function MapController() {
  const network = useStore(s => s.network)
  const map = useMap()
  React.useEffect(() => {
    if (network?.center) map.setView(network.center, network.zoom || 15)
  }, [network])
  return null
}

function MapClickHandler() {
  const network       = useStore(s => s.network)
  const sourceId      = useStore(s => s.sourceId)
  const targetId      = useStore(s => s.targetId)
  const epicenter     = useStore(s => s.epicenter)
  const disasterType  = useStore(s => s.disasterType)
  const blockModeActive = useStore(s => s.blockModeActive)
  const setSource     = useStore(s => s.setSource)
  const setTarget     = useStore(s => s.setTarget)
  const setEpicenter  = useStore(s => s.setEpicenter)
  const addBlockedRoad = useStore(s => s.addBlockedRoad)
  const addToast      = useStore(s => s.addToast)

  const findNearest = useCallback((lat, lon) => {
    if (!network?.nodes) return null
    let best = null, bestD = Infinity
    for (const n of network.nodes) {
      const d = (n.lat - lat) ** 2 + (n.lon - lon) ** 2
      if (d < bestD) { bestD = d; best = n }
    }
    return best
  }, [network])

  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng
      if (blockModeActive) {
        const n = findNearest(lat, lng)
        if (n) addBlockedRoad({ source: n.id, target: n.id, coords: [[n.lat, n.lon]] })
        return
      }
      if (sourceId === null) {
        const n = findNearest(lat, lng)
        if (n) { setSource(n.id); addToast({ type: 'success', title: 'Source Set', msg: `Node #${n.id}` }) }
      } else if (targetId === null) {
        const n = findNearest(lat, lng)
        if (n) { setTarget(n.id); addToast({ type: 'success', title: 'Target Set', msg: `Node #${n.id}` }) }
      } else if (disasterType !== 'none' && !epicenter) {
        setEpicenter([lat, lng])
        addToast({ type: 'info', title: 'Epicenter Placed', msg: `${lat.toFixed(4)}, ${lng.toFixed(4)}` })
      }
    }
  })
  return null
}

const LEGEND = [
  { color: '#3b82f6', label: 'Normal Route' },
  { color: '#10b981', label: 'Safe Route' },
  { color: '#ef4444', label: 'Blocked Road' },
  { color: '#f59e0b', label: 'Manual Block' },
  { color: '#6366f1', label: 'Vehicle' },
  { color: '#06b6d4', label: 'Search Wave' },
]

export default function MapView() {
  const network       = useStore(s => s.network)
  const sourceId      = useStore(s => s.sourceId)
  const targetId      = useStore(s => s.targetId)
  const epicenter     = useStore(s => s.epicenter)
  const disasterType  = useStore(s => s.disasterType)
  const disasterRadius = useStore(s => s.disasterRadius)
  const routeData     = useStore(s => s.routeData)
  const blockedRoads  = useStore(s => s.blockedRoads)
  const blockModeActive = useStore(s => s.blockModeActive)
  const simRunning    = useStore(s => s.simRunning)
  const simRadius     = useStore(s => s.simRadius)
  const simBlocked    = useStore(s => s.simBlocked)
  const addBlockedRoad = useStore(s => s.addBlockedRoad)

  const disasterColor = DISASTER_COLORS[disasterType] || '#ef4444'
  const activeRadius  = simRunning && simRadius > 0 ? simRadius : disasterRadius
  const activeBlocked = simRunning ? simBlocked : (routeData?.blocked_edges || [])

  const srcNode = network?.nodes?.find(n => n.id === sourceId)
  const tgtNode = network?.nodes?.find(n => n.id === targetId)

  return (
    <div className="map-area" style={{ cursor: blockModeActive ? 'crosshair' : 'default' }}>
      <MapContainer center={[18.5204, 73.8567]} zoom={13}
        style={{ height: '100%', width: '100%' }} zoomControl={true}>
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution="&copy; OpenStreetMap &copy; CartoDB"
          subdomains="abcd" maxZoom={19}
        />
        <MapController />
        <MapClickHandler />

        {/* Road Network */}
        {network?.edges?.map((edge, i) => {
          const blocked = blockedRoads.some(b =>
            (b.source === edge.source && b.target === edge.target) ||
            (b.source === edge.target && b.target === edge.source)
          )
          return (
            <Polyline key={i} positions={edge.coords}
              pathOptions={{
                color: blocked ? '#ef4444' : '#2d3f5e',
                weight: blocked ? 4 : 1.5,
                opacity: blocked ? 0.9 : 0.55,
                dashArray: blocked ? '6,4' : null,
              }}
              eventHandlers={{
                click: blockModeActive ? () => addBlockedRoad({ source: edge.source, target: edge.target, coords: edge.coords }) : undefined
              }}
            />
          )
        })}

        {/* Nodes (small) */}
        {network?.nodes?.slice(0, 2000).map(node => (
          <CircleMarker key={node.id} center={[node.lat, node.lon]} radius={2.5}
            pathOptions={{ color: '#334155', fillColor: '#475569', fillOpacity: 0.6, weight: 0.5 }}>
            <Popup><div style={{fontSize:12}}><b>Node #{node.id}</b><br/>{node.lat.toFixed(5)}, {node.lon.toFixed(5)}</div></Popup>
          </CircleMarker>
        ))}

        {/* Source marker */}
        {srcNode && (
          <CircleMarker center={[srcNode.lat, srcNode.lon]} radius={12}
            pathOptions={{ color: '#10b981', fillColor: '#10b981', fillOpacity: 0.85, weight: 2.5 }}>
            <Popup><b>🟢 Source #{sourceId}</b></Popup>
          </CircleMarker>
        )}

        {/* Target marker */}
        {tgtNode && (
          <CircleMarker center={[tgtNode.lat, tgtNode.lon]} radius={12}
            pathOptions={{ color: '#ef4444', fillColor: '#ef4444', fillOpacity: 0.85, weight: 2.5 }}>
            <Popup><b>🔴 Target #{targetId}</b></Popup>
          </CircleMarker>
        )}

        {/* Disaster circle */}
        {epicenter && disasterType !== 'none' && (
          <Circle center={epicenter} radius={activeRadius}
            pathOptions={{ color: disasterColor, fillColor: disasterColor, fillOpacity: 0.12, weight: 2, dashArray: '8,5' }} />
        )}

        {/* Blocked by disaster */}
        {activeBlocked.map((coords, i) => (
          <Polyline key={`db-${i}`} positions={coords}
            pathOptions={{ color: '#ef4444', weight: 4, opacity: 0.85 }} />
        ))}

        {/* Manually blocked roads */}
        {blockedRoads.map((road, i) => (
          <Polyline key={`mb-${i}`} positions={road.coords}
            pathOptions={{ color: '#f59e0b', weight: 5, opacity: 0.9, dashArray: '6,4' }} />
        ))}

        {/* Normal routes (blue dashes) */}
        {routeData?.normal_routes?.map((route, i) => (
          <Polyline key={`nr-${i}`} positions={route.path}
            pathOptions={{ color: ['#3b82f6','#8b5cf6','#ec4899'][i%3], weight: Math.max(2,5-i), opacity: Math.max(0.4, 0.9-i*0.15), dashArray:'12,8' }} />
        ))}
        {/* Fallback if no normal_routes array */}
        {!routeData?.normal_routes && routeData?.normal_route?.path && (
          <Polyline positions={routeData.normal_route.path}
            pathOptions={{ color: '#3b82f6', weight: 4, opacity: 0.85, dashArray: '12,8' }} />
        )}

        {/* Disaster-aware routes (green solid) */}
        {routeData?.disaster_routes?.map((route, i) => (
          <Polyline key={`dr-${i}`} positions={route.path}
            pathOptions={{ color: ['#10b981','#059669','#34d399'][i%3], weight: Math.max(3,5-i), opacity: Math.max(0.5, 0.95-i*0.1) }} />
        ))}
        {!routeData?.disaster_routes && routeData?.disaster_route?.path && (
          <Polyline positions={routeData.disaster_route.path}
            pathOptions={{ color: '#10b981', weight: 5, opacity: 0.9 }} />
        )}

        {/* Vehicle marker */}
        <VehicleMarker />

        {/* Search wave */}
        <SearchWave />
      </MapContainer>

      {/* Map legend */}
      <div className="map-legend">
        {LEGEND.map(({ color, label }) => (
          <div key={label} className="legend-item">
            <span className="legend-dot" style={{ background: color }} />
            {label}
          </div>
        ))}
      </div>

      {/* Block mode banner */}
      {blockModeActive && (
        <div className="block-mode-banner">
          🚧 Block Mode — Click any road on the map to close it
        </div>
      )}
    </div>
  )
}
