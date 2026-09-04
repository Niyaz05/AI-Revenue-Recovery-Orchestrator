import React from 'react'
import { ShieldCheck, Play, RefreshCw, Layers } from 'lucide-react'

export default function Navbar({ onTriggerClick, onRefresh, activeTab, setActiveTab }) {
  return (
    <header className="navbar">
      <div className="logo-group">
        <div className="logo-badge">RZP</div>
        <div>
          <h1 className="title-main">AI Revenue Recovery Orchestrator</h1>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Bounded Reinforcement & Safety Governance for Recurring Revenue (Test Mode)
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <button className="btn-secondary" onClick={onRefresh}>
          <RefreshCw size={14} style={{ marginRight: '0.4rem' }} /> Refresh
        </button>

        <button className="btn-primary" onClick={onTriggerClick}>
          <Play size={14} /> Run Test Scenario
        </button>
      </div>
    </header>
  )
}
