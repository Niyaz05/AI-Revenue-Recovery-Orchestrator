import React from 'react'

export default function ProvenanceBadge({ provenance }) {
  const getBadgeClass = (prov) => {
    switch (prov) {
      case 'SIMULATED':
        return 'badge-simulated'
      case 'SYNTHETIC_TRAINING_DATA':
        return 'badge-synthetic'
      case 'HELD_OUT_OFFLINE_EVAL':
        return 'badge-held-out'
      case 'RAZORPAY_TEST_MODE':
      default:
        return 'badge-razorpay-test'
    }
  }

  const getLabel = (prov) => {
    switch (prov) {
      case 'SIMULATED':
        return '⚡ Simulated'
      case 'SYNTHETIC_TRAINING_DATA':
        return '🧪 Synthetic Data'
      case 'HELD_OUT_OFFLINE_EVAL':
        return '📊 Held-Out Offline'
      case 'RAZORPAY_TEST_MODE':
        return '🔒 Razorpay Test Mode'
      default:
        return prov || 'Untagged'
    }
  }

  return (
    <span className={`badge-provenance ${getBadgeClass(provenance)}`}>
      {getLabel(provenance)}
    </span>
  )
}
