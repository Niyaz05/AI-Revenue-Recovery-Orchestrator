from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.models.base import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    init_db()


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["razorpay_mode"] == "TEST_MODE"


def test_dashboard_metrics():
    response = client.get("/api/dashboard/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "live_test_mode_metrics" in data
    assert "offline_evaluation_benchmark" in data


def test_trigger_scenario_and_list_endpoints():
    # Trigger a scenario
    scen_resp = client.post(
        "/api/test/trigger-scenario",
        json={"scenario_name": "temporary_bank_failure"},
    )
    assert scen_resp.status_code == 200
    scen_data = scen_resp.json()
    assert scen_data["status"] == "success"
    assert "action_id" in scen_data

    # List recovery actions
    actions_resp = client.get("/api/recovery-actions")
    assert actions_resp.status_code == 200
    actions = actions_resp.json()
    assert len(actions) >= 1

    # List subscriptions
    subs_resp = client.get("/api/subscriptions")
    assert subs_resp.status_code == 200
    subs = subs_resp.json()
    assert len(subs) >= 1

    # List audit logs
    audit_resp = client.get("/api/audit-log")
    assert audit_resp.status_code == 200
    audits = audit_resp.json()
    assert len(audits) >= 1


def test_simulate_event():
    resp = client.post(
        "/api/simulate-event",
        json={
            "customer_name": "Test Simulation Customer",
            "customer_email": "testsim@example.com",
            "amount": 7500.0,
            "failure_reason": "insufficient_funds",
            "plan_name": "Enterprise Plan",
            "is_opted_out": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "action_type" in data
    assert "explanation" in data


def test_customer_recovery_journey():
    # First trigger scenario so a customer exists
    scen_resp = client.post(
        "/api/test/trigger-scenario",
        json={"scenario_name": "high_value_customer"},
    )
    assert scen_resp.status_code == 200
    action_id = scen_resp.json()["action_id"]

    # Get customer id from actions list
    actions = client.get("/api/recovery-actions").json()
    cust_id = actions[0]["customer_id"]

    journey_resp = client.get(f"/api/customers/{cust_id}/recovery-journey")
    assert journey_resp.status_code == 200
    journey = journey_resp.json()
    assert "customer" in journey
    assert "timeline" in journey


def test_webhook_payment_link_paid_settles_action():
    # Create a payment-link recovery via simulate-event (insufficient funds -> PAYMENT_LINK)
    sim_resp = client.post(
        "/api/simulate-event",
        json={
            "customer_name": "Settle API Customer",
            "customer_email": "settleapi@example.com",
            "amount": 14999.0,
            "failure_reason": "insufficient_funds",
            "plan_name": "Pro Plan",
            "is_opted_out": False,
        },
    )
    assert sim_resp.status_code == 200
    action_id = sim_resp.json()["action_id"]

    # Post a payment_link.paid payload referencing that action
    payload = {
        "id": f"evt_api_{action_id}",
        "event": "payment_link.paid",
        "payload": {
            "payment_link": {
                "entity": {
                    "id": f"plink_api_{action_id}",
                    "amount": 1499900,
                    "currency": "INR",
                    "status": "paid",
                    "notes": {"recovery_action_id": str(action_id)},
                }
            },
            "payment": {
                "entity": {
                    "id": f"pay_api_{action_id}",
                    "amount": 1499900,
                    "currency": "INR",
                    "status": "captured",
                    "method": "upi",
                }
            },
        },
    }
    settle_resp = client.post("/api/webhooks/payment-link-paid", json={"payload": payload})
    assert settle_resp.status_code == 200
    data = settle_resp.json()
    assert data["status"] in ("settled", "skipped")

    # Idempotency: posting the same event again must not double-settle
    again = client.post("/api/webhooks/payment-link-paid", json={"payload": payload})
    assert again.json()["status"] == "skipped"
