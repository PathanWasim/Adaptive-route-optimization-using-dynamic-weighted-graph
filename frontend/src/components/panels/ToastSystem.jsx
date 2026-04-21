import React from 'react'
import { useEffect } from 'react'
import useStore from '../../store/useStore'

const ICONS = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' }

function Toast({ id, type, title, msg }) {
  const removeToast = useStore(s => s.removeToast)
  useEffect(() => {
    const t = setTimeout(() => removeToast(id), 4500)
    return () => clearTimeout(t)
  }, [id])
  return (
    <div className={`toast ${type}`}>
      <span className="toast-icon">{ICONS[type] || '💬'}</span>
      <div className="toast-body">
        <div className="toast-title">{title}</div>
        <div className="toast-msg">{msg}</div>
      </div>
      <button className="toast-close" onClick={() => removeToast(id)}>✕</button>
    </div>
  )
}

export default function ToastSystem() {
  const toasts = useStore(s => s.toasts)
  return (
    <div className="toast-container">
      {toasts.map(t => <Toast key={t.id} {...t} />)}
    </div>
  )
}
