import React from 'react'
import { CircleMarker } from 'react-leaflet'
import useStore from '../../store/useStore'

// Dijkstra = blue circle wave, A* = green directional beam
export default function SearchWave() {
  const waveNodes = useStore(s => s.waveNodes)

  return (
    <>
      {waveNodes.map((node, i) => {
        const isDijkstra = node.algo === 'dijkstra'
        const age = waveNodes.length - i  // older = more faded
        const opacity = Math.max(0.1, 1 - age * 0.04)
        return (
          <CircleMarker
            key={`wave-${i}`}
            center={[node.lat, node.lon]}
            radius={isDijkstra ? 5 : 6}
            pathOptions={{
              color: isDijkstra ? '#3b82f6' : '#10b981',
              fillColor: isDijkstra ? '#3b82f6' : '#10b981',
              fillOpacity: opacity * 0.7,
              opacity: opacity,
              weight: 1,
            }}
          />
        )
      })}
    </>
  )
}
