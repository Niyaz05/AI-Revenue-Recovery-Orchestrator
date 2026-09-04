import React, { useState } from 'react'
import ProvenanceBadge from './ProvenanceBadge'
import { FileText, ChevronRight, ChevronDown } from 'lucide-react'

export default function AuditLogViewer({ auditLogs = [] }) {
  const [expandedId, setExpandedId] = useState(null)

  const toggleExpand = (id) => {
    setExpandedId(expandedId === id ? null : id)
  }

  return (
    <div className="glass-panel" style={{ padding: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FileText size={20} color="var(--accent-purple)" />
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Immutable Governance & Audit Trail</h3>
        </div>
        <ProvenanceBadge provenance="RAZORPAY_TEST_MODE" />
      </div>

      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th style={{ width: '40px' }}></th>
              <th>Timestamp</th>
              <th>Event</th>
              <th>Action Executed</th>
              <th>Reasoning / Safety Grounds</th>
              <th>Provenance</th>
            </tr>
          </thead>
          <tbody>
            {auditLogs.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
                  No audit logs recorded yet.
                </td>
              </tr>
            ) : (
              auditLogs.map((log) => (
                <React.Fragment key={log.id}>
                  <tr onClick={() => toggleExpand(log.id)} style={{ cursor: 'pointer' }}>
                    <td>
                      {expandedId === log.id ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    </td>
                    <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {new Date(log.timestamp).toLocaleString()}
                    </td>
                    <td><code>{log.event_type}</code></td>
                    <td>
                      <span className="status-pill status-completed">{log.executed_action}</span>
                    </td>
                    <td style={{ fontSize: '0.85rem' }}>{log.reason}</td>
                    <td>
                      <ProvenanceBadge provenance={log.provenance} />
                    </td>
                  </tr>

                  {expandedId === log.id && (
                    <tr>
                      <td colSpan={6} style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem' }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                          <div>
                            <h4 style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                              AI Bandit Recommendation
                            </h4>
                            <pre style={{ fontSize: '0.75rem', background: '#070a10', padding: '0.75rem', borderRadius: '4px', overflowX: 'auto' }}>
                              {JSON.stringify(log.ai_recommendation, null, 2)}
                            </pre>
                          </div>
                          <div>
                            <h4 style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                              Safety Layer Policy Decision
                            </h4>
                            <pre style={{ fontSize: '0.75rem', background: '#070a10', padding: '0.75rem', borderRadius: '4px', overflowX: 'auto' }}>
                              {JSON.stringify(log.policy_decision, null, 2)}
                            </pre>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
