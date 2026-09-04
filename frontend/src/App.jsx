import React, { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import KPIOverview from './components/KPIOverview'
import BaselineComparisonChart from './components/BaselineComparisonChart'
import RecoveryJourneyDrilldown from './components/RecoveryJourneyDrilldown'
import LiveActionFeed from './components/LiveActionFeed'
import AuditLogViewer from './components/AuditLogViewer'
import ScenarioTriggerModal from './components/ScenarioTriggerModal'
import { LayoutDashboard, Users, History, Activity } from 'lucide-react'

export default function App() {
  const [activeTab, setActiveTab] = useState('overview')
  const [modalOpen, setModalOpen] = useState(false)
  const [metricsData, setMetricsData] = useState(null)
  const [actions, setActions] = useState([])
  const [auditLogs, setAuditLogs] = useState([])
  const [selectedCustomerId, setSelectedCustomerId] = useState(1)

  const loadData = async () => {
    try {
      // 1. Fetch dashboard metrics
      const mRes = await fetch('/api/dashboard/metrics')
      if (mRes.ok) {
        const mData = await mRes.json()
        setMetricsData(mData)
      }

      // 2. Fetch recovery actions
      const aRes = await fetch('/api/recovery-actions')
      if (aRes.ok) {
        const aData = await aRes.json()
        setActions(aData)
      }

      // 3. Fetch audit logs
      const lRes = await fetch('/api/audit-log')
      if (lRes.ok) {
        const lData = await lRes.json()
        setAuditLogs(lData)
      }
    } catch (err) {
      console.error('Failed to load dashboard data:', err)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  return (
    <div>
      <Navbar
        onTriggerClick={() => setModalOpen(true)}
        onRefresh={loadData}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      <main className="dashboard-container">
        {/* Navigation Tabs */}
        <div className="tabs-nav">
          <button
            className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            <LayoutDashboard size={16} style={{ display: 'inline', marginRight: '0.4rem' }} /> Overview & Baselines
          </button>
          <button
            className={`tab-button ${activeTab === 'journey' ? 'active' : ''}`}
            onClick={() => setActiveTab('journey')}
          >
            <Users size={16} style={{ display: 'inline', marginRight: '0.4rem' }} /> Customer Journey Drill-down
          </button>
          <button
            className={`tab-button ${activeTab === 'actions' ? 'active' : ''}`}
            onClick={() => setActiveTab('actions')}
          >
            <Activity size={16} style={{ display: 'inline', marginRight: '0.4rem' }} /> Live Interventions ({actions.length})
          </button>
          <button
            className={`tab-button ${activeTab === 'audit' ? 'active' : ''}`}
            onClick={() => setActiveTab('audit')}
          >
            <History size={16} style={{ display: 'inline', marginRight: '0.4rem' }} /> Governance & Audit Trail
          </button>
        </div>

        {/* Dynamic Views */}
        {activeTab === 'overview' && (
          <>
            <KPIOverview
              liveMetrics={metricsData?.live_test_mode_metrics}
              offlineBenchmark={metricsData?.offline_evaluation_benchmark}
            />
            <BaselineComparisonChart />
            <LiveActionFeed actions={actions.slice(0, 5)} />
          </>
        )}

        {activeTab === 'journey' && (
          <RecoveryJourneyDrilldown customerId={selectedCustomerId} />
        )}

        {activeTab === 'actions' && (
          <LiveActionFeed actions={actions} />
        )}

        {activeTab === 'audit' && (
          <AuditLogViewer auditLogs={auditLogs} />
        )}
      </main>

      <ScenarioTriggerModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onTriggered={loadData}
      />
    </div>
  )
}
