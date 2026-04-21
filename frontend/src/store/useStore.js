import { create } from 'zustand'

const useStore = create((set, get) => ({
  // ── Network State ──────────────────────────────────────────────────────────
  currentCity: null,
  network: null,           // { nodes, edges, stats, center, zoom }
  setNetwork: (city, data) => set({ currentCity: city, network: data }),

  // ── Selection State ────────────────────────────────────────────────────────
  sourceId: null,
  targetId: null,
  epicenter: null,         // [lat, lon]
  setSource: (id) => set({ sourceId: id }),
  setTarget: (id) => set({ targetId: id }),
  setEpicenter: (coords) => set({ epicenter: coords }),

  // ── Disaster Config ────────────────────────────────────────────────────────
  disasterType: 'none',
  disasterRadius: 200,
  disasterSeverity: 0.6,
  setDisasterType: (t) => set({ disasterType: t }),
  setDisasterRadius: (r) => set({ disasterRadius: r }),
  setDisasterSeverity: (s) => set({ disasterSeverity: s }),

  // ── Algorithm Config ───────────────────────────────────────────────────────
  algorithm: 'dijkstra',
  kPaths: 3,
  animateAlgo: true,
  algoSpeed: 80,
  setAlgorithm: (a) => set({ algorithm: a }),
  setKPaths: (k) => set({ kPaths: k }),
  setAnimateAlgo: (v) => set({ animateAlgo: v }),
  setAlgoSpeed: (s) => set({ algoSpeed: s }),

  // ── Objective Weights ──────────────────────────────────────────────────────
  alpha: 1.0,
  beta: 1.0,
  gamma: 1.0,
  setAlpha: (v) => set({ alpha: v }),
  setBeta: (v) => set({ beta: v }),
  setGamma: (v) => set({ gamma: v }),

  // ── Route Results ──────────────────────────────────────────────────────────
  routeData: null,         // full response from /api/compute_route
  routeHistory: [],
  setRouteData: (data) => set((s) => ({
    routeData: data,
    routeHistory: [
      { ts: new Date().toLocaleTimeString(), algo: data.algorithm, dist: data.normal_route?.distance, disaster_dist: data.disaster_route?.distance },
      ...s.routeHistory.slice(0, 7),
    ]
  })),

  // ── Algorithm Visualization Steps ─────────────────────────────────────────
  algoSteps: [],
  currentStep: 0,
  isAnimating: false,
  setAlgoSteps: (steps) => set({ algoSteps: steps, currentStep: 0 }),
  setCurrentStep: (n) => set({ currentStep: n }),
  setIsAnimating: (v) => set({ isAnimating: v }),

  // ── Feature 1: Disaster Simulation ────────────────────────────────────────
  simRunning: false,
  simRadius: 0,
  simBlocked: [],
  simPercent: 0,
  setSimRunning: (v) => set({ simRunning: v }),
  updateSimTick: ({ radius, blocked_edges, percent }) => set({ simRadius: radius, simBlocked: blocked_edges, simPercent: percent }),

  // ── Feature 2: Blocked Roads ───────────────────────────────────────────────
  blockedRoads: [],        // [{ source, target, coords }]
  blockModeActive: false,
  addBlockedRoad: (road) => set((s) => ({ blockedRoads: [...s.blockedRoads, road] })),
  clearBlockedRoads: () => set({ blockedRoads: [] }),
  setBlockMode: (v) => set({ blockModeActive: v }),

  // ── Feature 3: Vehicle Animation ──────────────────────────────────────────
  vehiclePath: [],         // [[lat,lon], ...]
  vehiclePos: null,        // current [lat, lon]
  vehicleProgress: 0,      // 0..1
  vehicleRunning: false,
  setVehiclePath: (path) => set({ vehiclePath: path, vehiclePos: path[0] || null, vehicleProgress: 0 }),
  updateVehicle: (pos, pct) => set({ vehiclePos: pos, vehicleProgress: pct }),
  setVehicleRunning: (v) => set({ vehicleRunning: v }),

  // ── Feature 4: Search Wave ─────────────────────────────────────────────────
  waveNodes: [],           // [{ lat, lon, algo, step }]
  waveRunning: false,
  waveStepCount: 0,
  appendWaveNode: (node) => set((s) => ({ waveNodes: [...s.waveNodes, node], waveStepCount: s.waveStepCount + 1 })),
  clearWave: () => set({ waveNodes: [], waveStepCount: 0 }),
  setWaveRunning: (v) => set({ waveRunning: v }),

  // ── UI State ───────────────────────────────────────────────────────────────
  activeTab: 'city',
  sidebarOpen: true,
  toasts: [],
  setActiveTab: (t) => set({ activeTab: t }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  addToast: (toast) => set((s) => ({ toasts: [...s.toasts, { id: Date.now(), ...toast }] })),
  removeToast: (id) => set((s) => ({ toasts: s.toasts.filter(t => t.id !== id) })),

  // ── Loading State ──────────────────────────────────────────────────────────
  loadingNetwork: false,
  computingRoute: false,
  setLoadingNetwork: (v) => set({ loadingNetwork: v }),
  setComputingRoute: (v) => set({ computingRoute: v }),

  // ── Topbar Stats ───────────────────────────────────────────────────────────
  tbNodes: '—',
  tbEdges: '—',
  setTopbarStats: (nodes, edges) => set({ tbNodes: nodes, tbEdges: edges }),

  // ── Reset ──────────────────────────────────────────────────────────────────
  showHelpModal: true, // auto show on first load
  setShowHelpModal: (v) => set({ showHelpModal: v }),
  resetSelections: () => set({
    sourceId: null, targetId: null, epicenter: null,
    routeData: null, algoSteps: [], waveNodes: [],
    blockedRoads: [], vehiclePath: [], vehiclePos: null,
    simRunning: false, simRadius: 0, simBlocked: [],
  }),
}))

export default useStore
