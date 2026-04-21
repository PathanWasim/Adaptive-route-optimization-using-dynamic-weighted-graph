import React from 'react'
import useStore from '../../store/useStore'

const TYPES = [
  { value: 'none',       icon: '🚫', label: 'None' },
  { value: 'fire',       icon: '🔥', label: 'Fire' },
  { value: 'flood',      icon: '🌊', label: 'Flood' },
  { value: 'earthquake', icon: '⚡', label: 'Quake' },
]

function RangeSlider({ label, value, min, max, step, onChange, unit = '', cls = 'danger' }) {
  return (
    <div className="range-wrap">
      <div className="range-header">
        <span className="range-label">{label}</span>
        <span className="range-val">{value}{unit}</span>
      </div>
      <input type="range" className={cls} min={min} max={max} step={step}
        value={value} onChange={e => onChange(parseFloat(e.target.value))} />
    </div>
  )
}

export default function DisasterPanel() {
  const disasterType     = useStore(s => s.disasterType)
  const disasterRadius   = useStore(s => s.disasterRadius)
  const disasterSeverity = useStore(s => s.disasterSeverity)
  const epicenter        = useStore(s => s.epicenter)
  const { setDisasterType, setDisasterRadius, setDisasterSeverity } = useStore.getState()

  return (
    <>
      <div className="section-title">Disaster Scenario</div>

      <div>
        <label className="field-label">Disaster Type</label>
        <div className="disaster-grid">
          {TYPES.map(({ value, icon, label }) => (
            <button
              key={value}
              className={`disaster-btn ${disasterType === value ? 'active' : ''}`}
              onClick={() => setDisasterType(value)}
            >
              <span className="d-icon">{icon}</span>
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>

      {disasterType !== 'none' && (
        <>
          <hr className="divider" />
          <RangeSlider label="Effect Radius" value={disasterRadius} min={50} max={800} step={10}
            onChange={setDisasterRadius} unit="m" />
          <RangeSlider label="Severity Level" value={disasterSeverity} min={0.1} max={1.0} step={0.05}
            onChange={setDisasterSeverity} />

          <div className={`info-box ${epicenter ? 'success' : 'warning'}`}>
            {epicenter
              ? `✓ Epicenter set: ${epicenter[0].toFixed(4)}, ${epicenter[1].toFixed(4)}`
              : '📍 Click on the map to place the disaster epicenter'}
          </div>

          <div className="info-box info" style={{ fontSize: 10 }}>
            The disaster model applies hazard multipliers to road weights within the radius.
            Severity 1.0 = maximum block probability.
          </div>
        </>
      )}

      {disasterType === 'none' && (
        <div className="empty-state">
          <div className="em-icon">🌍</div>
          Select a disaster type above to configure the hazard scenario
        </div>
      )}
    </>
  )
}
