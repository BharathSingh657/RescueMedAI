"""
Unit tests for Medical Decision Support Triage module.
Verifies rules, Bayesian inference, hybrid conservative escalation, and safety notices.
"""

import pytest
from src.backend.state import PatientRecord, MEDICAL_SAFETY_DISCLAIMER
from src.backend.triage.rules import EmergencyRuleEngine
from src.backend.triage.bayesian import EmergencyBayesianNetwork
from src.backend.triage.arbitrator import HybridTriageArbitrator


def test_rule_engine_critical_hypoxemia():
    engine = EmergencyRuleEngine()
    patient = PatientRecord(
        patient_id="PT-001",
        age=42,
        systolic_bp=110,
        diastolic_bp=70,
        spo2=84,  # Critical hypoxemia < 88%
        heart_rate=98,
        respiratory_rate=26,
        consciousness="Alert",
        symptoms=["Shortness of breath"],
        visible_trauma="Chest wall bruising",
        ambulatory=False
    )
    res = engine.evaluate(patient)
    assert res.suggested_priority == "RED"
    assert any(r["rule_id"] == "RULE-RESP-01" for r in res.activated_rules)


def test_bayesian_network_shock_case():
    net = EmergencyBayesianNetwork()
    patient = PatientRecord(
        patient_id="PT-002",
        age=28,
        systolic_bp=75,    # Severe hypotension
        diastolic_bp=45,
        spo2=91,
        heart_rate=142,    # Tachycardia / critical
        respiratory_rate=28,
        consciousness="Voice",
        symptoms=["Massive External Bleeding"],
        visible_trauma="Severe thigh laceration",
        ambulatory=False
    )
    res = net.infer(patient)
    # Sum of probabilities must equal 1.0 (approx)
    total_prob = sum(res.hypotheses_posteriors.values())
    assert pytest.approx(total_prob, rel=1e-3) == 1.0

    # Hypovolemic Shock should be highest
    assert res.top_condition == "Hypovolemic / Hemorrhagic Shock"
    assert res.top_probability > 0.40
    assert res.entropy >= 0.0


def test_hybrid_arbitrator_conservative_escalation():
    arbitrator = HybridTriageArbitrator()

    # Borderline patient: rules might classify as Yellow, but Bayesian identifies high critical risk
    borderline_patient = PatientRecord(
        patient_id="PT-003",
        age=65,
        systolic_bp=86,    # Low
        diastolic_bp=55,
        spo2=89,           # Low
        heart_rate=128,    # Tachycardia
        respiratory_rate=24,
        consciousness="Voice",
        symptoms=["Chest tightness", "Difficulty breathing"],
        visible_trauma="None",
        ambulatory=False
    )
    decision = arbitrator.arbitrate(borderline_patient)

    # Must be either RED (escalated) or YELLOW
    assert decision.priority in ["RED", "YELLOW"]
    assert decision.disclaimer == MEDICAL_SAFETY_DISCLAIMER
    assert decision.confidence_score > 0.0


def test_stable_patient_green():
    arbitrator = HybridTriageArbitrator()
    stable_patient = PatientRecord(
        patient_id="PT-004",
        age=22,
        systolic_bp=120,
        diastolic_bp=80,
        spo2=98,
        heart_rate=72,
        respiratory_rate=16,
        consciousness="Alert",
        symptoms=["Minor arm abrasion"],
        visible_trauma="Scrape",
        ambulatory=True
    )
    decision = arbitrator.arbitrate(stable_patient)
    assert decision.priority == "GREEN"
    assert decision.priority_level.startswith("Delayed")
    assert decision.bayesian_result.top_condition == "Stable Trauma / Minor Injury"
