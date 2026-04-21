import { useEffect, useRef } from 'react'
import { io } from 'socket.io-client'
import useStore from '../store/useStore'

let socket = null

export function useSocket() {
  const updateSimTick = useStore(s => s.updateSimTick)
  const setSimRunning = useStore(s => s.setSimRunning)
  const addToast = useStore(s => s.addToast)
  const initialized = useRef(false)

  useEffect(() => {
    if (initialized.current) return
    initialized.current = true

    socket = io('/', { path: '/socket.io', transports: ['websocket', 'polling'] })

    socket.on('connect', () => {
      console.log('[Socket] Connected:', socket.id)
    })
    socket.on('disconnect', () => {
      console.log('[Socket] Disconnected')
    })
    socket.on('disaster_tick', (data) => {
      updateSimTick(data)
      if (data.at_max) {
        setSimRunning(false)
        addToast({ type: 'warning', title: 'Max Spread Reached', msg: `Disaster reached radius ${data.radius}m` })
      }
    })
    socket.on('sim_started', () => {
      addToast({ type: 'info', title: 'Simulation Started', msg: 'Disaster is expanding in real-time via WebSocket' })
    })
    socket.on('sim_stopped', () => {
      addToast({ type: 'success', title: 'Simulation Stopped', msg: 'Disaster expansion halted' })
    })

    return () => {
      socket?.disconnect()
      initialized.current = false
    }
  }, [])

  const startSim = (data) => {
    socket?.emit('start_disaster_sim', data)
  }
  const stopSim = () => {
    socket?.emit('stop_disaster_sim')
  }

  return { startSim, stopSim }
}

export { socket }
