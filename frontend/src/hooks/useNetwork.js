import useStore from '../store/useStore'

const BASE = '/api'

export function useNetwork() {
  async function fetchCities() {
    const res = await fetch(`${BASE}/cities`)
    if (!res.ok) throw new Error('Failed to fetch cities')
    const data = await res.json()
    return data.cities   // [{key, name, center, zoom}]
  }

  async function loadNetwork(cityKey) {
    const { setLoadingNetwork, setNetwork, setTopbarStats, addToast } = useStore.getState()
    setLoadingNetwork(true)
    try {
      const res = await fetch(`${BASE}/load_network`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ city_key: cityKey }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Load failed')
      setNetwork(cityKey, data)
      setTopbarStats(
        data.stats?.num_nodes?.toLocaleString?.() ?? data.stats?.num_nodes,
        data.stats?.num_edges?.toLocaleString?.() ?? data.stats?.num_edges
      )
      addToast({ type: 'success', title: 'Network Loaded', msg: `${data.stats.num_nodes} nodes · ${data.stats.num_edges} edges` })
      return data
    } catch (e) {
      useStore.getState().addToast({ type: 'error', title: 'Load Failed', msg: e.message })
      return null
    } finally {
      setLoadingNetwork(false)
    }
  }

  async function computeRoute(payload) {
    const { setComputingRoute, setRouteData, setAlgoSteps, addToast } = useStore.getState()
    setComputingRoute(true)
    try {
      const res = await fetch(`${BASE}/compute_route`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Compute failed')

      // Normalise: backend always returns `normal_route` and `normal_routes`
      setRouteData(data)

      // Feed animation steps if present
      const steps = data.normal_route?.steps
      if (steps && Array.isArray(steps)) setAlgoSteps(steps)

      addToast({
        type: 'success',
        title: 'Route Computed',
        msg: `${data.algorithm} · ${data.normal_route?.distance?.toFixed(0)}m  (${(data.normal_route?.computation_time * 1000).toFixed(1)}ms)`,
      })
      return data
    } catch (e) {
      useStore.getState().addToast({ type: 'error', title: 'Compute Failed', msg: e.message })
      return null
    } finally {
      setComputingRoute(false)
    }
  }

  async function blockRoad(payload) {
    const res = await fetch(`${BASE}/block_road`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    return await res.json()
  }

  async function runBenchmark(sizes = [50, 100, 200, 500, 1000]) {
    const res = await fetch(`${BASE}/benchmark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sizes, runs_per_size: 3 }),
    })
    if (!res.ok) throw new Error('Benchmark failed')
    return await res.json()
  }

  async function saveVisualization(payload) {
    const res = await fetch(`${BASE}/save_visualization`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    return await res.json()
  }

  return { fetchCities, loadNetwork, computeRoute, blockRoad, runBenchmark, saveVisualization }
}
