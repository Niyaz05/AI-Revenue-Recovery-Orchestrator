import React from 'react'
import ProvenanceBadge from './ProvenanceBadge'
import { Activity, ExternalLink, ShieldAlert, ShieldCheck } from 'lucide-react'

export default function LiveActionFeed({ actions = [] }) {
  return (
    <div className="glass-panel" style={{ padding: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity size={20} color="var(--accent-emerald)" />
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Live Recovery Interventions</h3>
        </div>
        <ProvenanceBadge provenance="RAZORPAY_TEST_MODE" />
      </div>

      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Customer</th>
              <th>Action Type</th>
              <th>Safety Status</th>
              <th>Execution Details</th>
              <th>Recovered</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {actions.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
                  No live recovery actions yet. Trigger a test scenario above!
                </td>
              </tr>
            ) : (
              actions.map((act) => (
                <tr key={act.id}>
                  <td><code>#{act.id}</code></td>
                  <td>
                    <div>{act.customer_name}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{act.customer_email}</div>
                  </td>
                  <td>
                    <span className="status-pill status-completed">{act.action_type}</span>
                  </td>
                  <td>
                    {act.blocked_by_policy ? (
                      <span className="status-pill status-blocked" style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                        <ShieldAlert size={12} /> Blocked
                      </span>
                    ) : (
                      <span className="status-pill status-completed" style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                        <ShieldCheck size={12} /> Approved
                      </span>
                    )}
                  </td>
                  <td>
                    {act.razorpay_payment_link_id ? (
                      <a
                        href={`https://rzp.io/i/${act.razorpay_payment_link_id}`}
                        target="_blank"
                        rel="noreferrer"
                        style={{ color: 'var(--accent-blue)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
                      >
                        <code>{act.razorpay_payment_link_id}</code> <ExternalLink size={12} />
                      </a>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>Internal Schedule</span>
                    )}
                  </td>
                  <td style={{ color: act.recovered_amount > 0 ? 'var(--accent-emerald)' : 'var(--text-secondary)', fontWeight: act.recovered_amount > 0 ? 700 : 400 }}>
                    {act.recovered_amount > 0 ? `₹${act.recovered_amount.toLocaleString('en-IN')}` : '—'}
                  </td>
                  <td style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    {act.created_at ? new Date(act.created_at).toLocaleTimeString() : 'Just now'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
