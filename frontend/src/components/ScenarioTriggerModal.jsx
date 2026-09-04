import React, { useState } from 'react'
import { Play, X, CheckCircle, AlertCircle } from 'lucide-react'

export default function ScenarioTriggerModal({ isOpen, onClose, onTriggered }) {
  const [selectedScenario, setSelectedScenario] = useState('temporary_bank_failure')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  if (!isOpen) return null

  const scenarios = [
    { id: 'temporary_bank_failure', name: '1. Temporary Bank Failure (Gateway Cooldown)', desc: 'AI observes glitch -> schedules WAIT -> succeeds on retry' },
    { id: 'insufficient_funds', name: '2. Insufficient Funds (Payment Link Recovery)', desc: 'AI generates multichannel Razorpay Payment Link' },
    { id: 'invalid_payment_method', name: '3. Invalid Payment Method / Expired Card', desc: 'AI requests updated payment method mandate' },
    { id: 'high_value_customer', name: '4. High-Value Customer (> ₹50,000)', desc: 'Safety layer routes to white-glove human billing escalation' },
    { id: 'opted_out_customer', name: '5. Opted-Out Customer (Hard Safety Veto)', desc: 'Safety layer strictly vetoes intervention to STOP_RECOVERY' },
  ]

  const handleRun = async () => {
    setLoading(true)
    setResult(null)
    try {
      const res = await fetch('/api/test/trigger-scenario', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario_name: selectedScenario }),
      })
      const data = await res.json()
      setResult(data)
      if (onTriggered) onTriggered()
    } catch (err) {
      setResult({ status: 'error', message: err.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.75)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
    }}>
      <div className="glass-panel" style={{ width: '550px', padding: '2rem', background: '#111827', position: 'relative' }}>
        <button
          onClick={onClose}
          style={{ position: 'absolute', top: '1.25rem', right: '1.25rem', background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}
        >
          <X size={20} />
        </button>

        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.5rem' }}>
          Execute Test Mode Recovery Scenario
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
          Select a canonical scenario to simulate through the full webhook & safety pipeline.
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '1.5rem' }}>
          {scenarios.map((s) => (
            <label
              key={s.id}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.75rem',
                padding: '0.75rem',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-color)',
                background: selectedScenario === s.id ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
                borderColor: selectedScenario === s.id ? 'var(--accent-blue)' : 'var(--border-color)',
                cursor: 'pointer',
              }}
            >
              <input
                type="radio"
                name="scenario"
                value={s.id}
                checked={selectedScenario === s.id}
                onChange={() => setSelectedScenario(s.id)}
                style={{ marginTop: '0.25rem' }}
              />
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{s.name}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{s.desc}</div>
              </div>
            </label>
          ))}
        </div>

        {result && (
          <div style={{
            padding: '1rem',
            borderRadius: 'var(--radius-sm)',
            marginBottom: '1.25rem',
            background: result.status === 'success' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)',
            border: `1px solid ${result.status === 'success' ? 'var(--accent-emerald)' : 'var(--accent-rose)'}`,
          }}>
            <div style={{ fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.3rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              {result.status === 'success' ? <CheckCircle size={16} color="var(--accent-emerald)" /> : <AlertCircle size={16} color="var(--accent-rose)" />}
              Execution Result: {result.action_type} ({result.action_status})
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {result.explanation}
            </p>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
          <button className="btn-secondary" onClick={onClose}>Close</button>
          <button className="btn-primary" onClick={handleRun} disabled={loading}>
            <Play size={14} /> {loading ? 'Executing...' : 'Trigger Pipeline'}
          </button>
        </div>
      </div>
    </div>
  )
}
