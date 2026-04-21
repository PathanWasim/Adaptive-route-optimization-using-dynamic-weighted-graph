import React, { useEffect, useRef } from 'react'
import L from 'leaflet'
import { useMap } from 'react-leaflet'
import useStore from '../../store/useStore'

const vehicleIcon = L.divIcon({
  html: '<div class="vehicle-marker-icon">🚗</div>',
  className: '',
  iconSize: [28, 28],
  iconAnchor: [14, 14],
})

export default function VehicleMarker() {
  const map = useMap()
  const vehiclePos = useStore(s => s.vehiclePos)
  const vehiclePath = useStore(s => s.vehiclePath)
  const vehicleRunning = useStore(s => s.vehicleRunning)
  const updateVehicle = useStore(s => s.updateVehicle)
  const setVehicleRunning = useStore(s => s.setVehicleRunning)
  const markerRef = useRef(null)
  const rafRef = useRef(null)
  const stepRef = useRef(0)

  // Create or update marker
  useEffect(() => {
    if (!vehiclePos) {
      if (markerRef.current) { markerRef.current.remove(); markerRef.current = null }
      return
    }
    if (!markerRef.current) {
      markerRef.current = L.marker(vehiclePos, { icon: vehicleIcon, zIndexOffset: 1000 }).addTo(map)
    } else {
      markerRef.current.setLatLng(vehiclePos)
    }
  }, [vehiclePos, map])

  // Animate vehicle along path using requestAnimationFrame
  useEffect(() => {
    if (!vehicleRunning || vehiclePath.length < 2) return

    const totalPoints = vehiclePath.length
    stepRef.current = 0
    const SPEED_MS = 40 // ms per segment

    let lastTime = null
    let accumulated = 0

    function frame(ts) {
      if (!lastTime) lastTime = ts
      const delta = ts - lastTime
      lastTime = ts
      accumulated += delta

      while (accumulated >= SPEED_MS && stepRef.current < totalPoints - 1) {
        accumulated -= SPEED_MS
        stepRef.current++
        const pos = vehiclePath[stepRef.current]
        const pct = stepRef.current / (totalPoints - 1)
        updateVehicle(pos, pct)
      }

      if (stepRef.current < totalPoints - 1) {
        rafRef.current = requestAnimationFrame(frame)
      } else {
        setVehicleRunning(false)
      }
    }

    rafRef.current = requestAnimationFrame(frame)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [vehicleRunning, vehiclePath])

  return null
}
