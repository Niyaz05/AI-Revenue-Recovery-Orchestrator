import React, { useState, useEffect } from 'react'
import ProvenanceBadge from './ProvenanceBadge'
import { User, Activity, AlertTriangle, ShieldCheck, CheckCircle2, Link2, Clock } from 'lucide-react'

export default function RecoveryJourneyDrilldown({ customerId = 1 }) {
  const [journeyData, setJourneyData] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchJourney(customerId)
  }, [customerId])

  const fetchJourney = async (id) => {
    setLoading(true)
    try {
      const res = await fetch(`/api/customers/${id}/recovery-journey`)
      if (res.ok) {
        const data = await res.json()
        setJourneyData(data)
      } else {
        // Mock fallback journey for UI demonstration
        setJourneyData(getFallbackJourney(id))
      }
    } catch (e) {
      setJourneyData(getFallbackJourney(id))
    } finally {
      setLoading(false)
    }
  }

  const getFallbackJourney = (id) => ({
    customer: {
      id,
      name: 'Vikram Mehta',
      email: 'vikram@enterprise.in',
      phone: '+91 98765 43210',
      lifetime_value: 120000,
      opted_out: false,
      failure_rate: 0.08,
    },
    timeline: [
      {
        id: 101,
        event_type: 'subscription.pending',
        timestamp: '2026-08-24T14:30:00Z',
        executed_action: 'WAIT',
        reason: 'Temporary bank gateway downtime detected. Waiting 4h cooldown before retrying.',
        ai_recommendation: { score: 0.72, confidence: 0.85, action: 'WAIT' },
        policy_decision: { allowed_actions: ['WAIT', 'RETRY', 'PAYMENT_LINK'], must_escalate: false },
        provenance: 'RAZORPAY_TEST_MODE',
      },
      {
        id: 102,
        event_type: 'payment.failed',
        timestamp: '2026-08-24T18:35:00Z',
        executed_action: 'PAYMENT_LINK',
        reason: 'Insufficient balance reported on card. LinUCB generated multichannel Razorpay Payment Link.',
        ai_recommendation: { score: 0.88, confidence: 0.91, action: 'PAYMENT_LINK' },
        policy_decision: { allowed_actions: ['PAYMENT_LINK', 'SEND_REMINDER'], must_escalate: false },
        provenance: 'RAZORPAY_TEST_MODE',
      },
      {
        id: 103,
        event_type: 'payment_link.paid',
        timestamp: '2026-08-24T20:15:00Z',
        executed_action: 'RECOVERY_RESOLVED',
        recovered_amount: 14999.0,
        reason: 'Customer successfully settled ₹14,999.00 via UPI Payment Link. Subscription reactivated.',
        provenance: 'RAZORPAY_TEST_MODE',
      },
    ],
  })

  if (!journeyData) return null

  return (
    <div className="glass-panel" style={{ padding: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <User size={22} color="var(--accent-blue)" />
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>
              Customer Recovery Journey: {journeyData.customer.name}
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {journeyData.customer.email} • LTV: ₹{journeyData.customer.lifetime_value.toLocaleString('en-IN')}
            </p>
          </div>
        </div>
        <ProvenanceBadge provenance="RAZORPAY_TEST_MODE" />
      </div>

      {/* Timeline Steps */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', borderLeft: '2px solid rgba(255,255,255,0.1)', paddingLeft: '1.5rem', marginLeft: '0.75rem' }}>
        {journeyData.timeline.map((step, idx) => (
          <div key={idx} style={{ position: 'relative' }}>
            <div style={{
              position: 'absolute',
              left: '-2.05rem',
              top: '0.2rem',
              width: '14px',
              height: '14px',
              borderRadius: '50%',
              background: step.recovered_amount ? 'var(--accent-emerald)' : 'var(--accent-blue)',
              border: '2px solid var(--bg-primary)'
            }} />

            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className="status-pill status-completed">{step.executed_action}</span>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{step.event_type}</span>
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  <Clock size={12} style={{ display: 'inline', marginRight: '0.25rem' }} />
                  {new Date(step.timestamp).toLocaleTimeString()}
                </span>
              </div>

              <p style={{ fontSize: '0.85rem', color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
                {step.reason}
              </p>

              {step.recovered_amount && (
                <div style={{ color: 'var(--accent-emerald)', fontWeight: 700, fontSize: '0.9rem' }}>
                  🎉 Recovered Revenue: ₹{step.recovered_amount.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
