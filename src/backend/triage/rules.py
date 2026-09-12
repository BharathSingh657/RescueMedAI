"""
Explicit Rule Engine for Emergency Decision Support (Prototype/Educational).
Encodes transparent triage escalation rules inspired by educational START/ESI frameworks.

SAFETY NOTICE:
AI-Assisted Emergency Decision Support — Prototype — Human Clinical Review Required.
This system provides educational decision support and does not provide clinical diagnosis,
autonomous medical triage, or validated therapeutic directives.
"""

from typing import List, Dict, Any
from src.backend.state import PatientRecord, RuleEvaluationResult, MEDICAL_SAFETY_DISCLAIMER


class EmergencyRuleEngine:
    def __init__(self):
        self.disclaimer = MEDICAL_SAFETY_DISCLAIMER

    def evaluate(self, patient: PatientRecord) -> RuleEvaluationResult:
        """
        Evaluates explicit deterministic safety rules against patient vitals and symptoms.
        Applies conservative escalation: the most urgent triggered condition determines suggested priority.
        """
        activated: List[Dict[str, Any]] = []

        # Rule 1: Ambulatory check (START triage initial filter)
        if patient.ambulatory and patient.spo2 >= 94 and patient.systolic_bp >= 100:
            activated.append({
                "rule_id": "RULE-AMB-01",
                "name": "Ambulatory Patient with Stable Base Vitals",
                "priority": "GREEN",
                "condition": f"Ambulatory=True, SpO2={patient.spo2}%, Systolic BP={patient.systolic_bp} mmHg",
                "rationale": "Patient is ambulatory and demonstrating stable baseline oxygenation and perfusion."
            })

        # Rule 2: Critical Respiratory Compromise
        if patient.spo2 < 88 or patient.respiratory_rate < 8 or patient.respiratory_rate > 32:
            activated.append({
                "rule_id": "RULE-RESP-01",
                "name": "Severe Respiratory Compromise",
                "priority": "RED",
                "condition": f"SpO2={patient.spo2}%, Respiratory Rate={patient.respiratory_rate}/min",
                "rationale": "Extreme hypoxemia or respiratory failure pattern requires immediate field airway/oxygen support."
            })
        elif 88 <= patient.spo2 < 93 or 25 <= patient.respiratory_rate <= 32:
            activated.append({
                "rule_id": "RULE-RESP-02",
                "name": "Moderate Respiratory Distress",
                "priority": "YELLOW",
                "condition": f"SpO2={patient.spo2}%, Respiratory Rate={patient.respiratory_rate}/min",
                "rationale": "Sub-optimal oxygenation or tachypnea requires urgent observation."
            })

        # Rule 3: Circulatory Collapse / Hypotension
        if patient.systolic_bp < 85 or patient.heart_rate > 135 or (patient.heart_rate < 45 and patient.consciousness != "Alert"):
            activated.append({
                "rule_id": "RULE-CIRC-01",
                "name": "Severe Hemodynamic Instability / Decompensated Shock",
                "priority": "RED",
                "condition": f"Systolic BP={patient.systolic_bp} mmHg, Heart Rate={patient.heart_rate} bpm",
                "rationale": "Profund hypotension or extreme heart rate indicative of cardiovascular compromise."
            })
        elif 85 <= patient.systolic_bp < 100 or 110 <= patient.heart_rate <= 135:
            activated.append({
                "rule_id": "RULE-CIRC-02",
                "name": "Compensated Circulatory Stress",
                "priority": "YELLOW",
                "condition": f"Systolic BP={patient.systolic_bp} mmHg, Heart Rate={patient.heart_rate} bpm",
                "rationale": "Borderline blood pressure or pronounced tachycardia requiring urgent monitoring."
            })

        # Rule 4: Neurological Impairment (AVPU Scale)
        if patient.consciousness in ["Pain", "Unresponsive"]:
            activated.append({
                "rule_id": "RULE-NEURO-01",
                "name": "Severe Neurological Depression",
                "priority": "RED",
                "condition": f"Consciousness='{patient.consciousness}'",
                "rationale": "Inability to maintain airway or severe central nervous system depression."
            })
        elif patient.consciousness == "Voice":
            activated.append({
                "rule_id": "RULE-NEURO-02",
                "name": "Altered Mental Status (Responds to Voice)",
                "priority": "YELLOW",
                "condition": "Consciousness='Voice'",
                "rationale": "Lethargy or disorientation warranting urgent assessment."
            })

        # Rule 5: Critical High-Risk Injury Symptoms
        critical_symptoms = {
            "Massive External Bleeding",
            "Penetrating Chest Trauma",
            "Severe Smoke Inhalation",
            "Crush Injury (Prolonged)"
        }
        present_critical = [s for s in patient.symptoms if s in critical_symptoms]
        if present_critical:
            activated.append({
                "rule_id": "RULE-TRAUMA-01",
                "name": "High-Acuity Trauma Flag",
                "priority": "RED",
                "condition": f"Symptoms: {', '.join(present_critical)}",
                "rationale": "Identified high-acuity physical hazard associated with rapid physiological decompensation."
            })

        # Rule 6: Urgent Non-Immediate Symptoms
        urgent_symptoms = {
            "Suspected Limb Fracture",
            "Moderate Burns (<20%)",
            "Severe Laceration (Controlled)",
            "Blunt Abdominal Contusion"
        }
        present_urgent = [s for s in patient.symptoms if s in urgent_symptoms]
        if present_urgent:
            activated.append({
                "rule_id": "RULE-TRAUMA-02",
                "name": "Urgent Trauma Presentation",
                "priority": "YELLOW",
                "condition": f"Symptoms: {', '.join(present_urgent)}",
                "rationale": "Significant injury requiring hospital treatment but without immediate airway/circulatory collapse."
            })

        # Priority arbitration (conservative escalation)
        has_red = any(r["priority"] == "RED" for r in activated)
        has_yellow = any(r["priority"] == "YELLOW" for r in activated)

        if has_red:
            suggested_priority = "RED"
            escalation_triggered = True
            rationale = "Immediate priority: One or more life-critical physiological or trauma thresholds breached."
        elif has_yellow:
            suggested_priority = "YELLOW"
            escalation_triggered = False
            rationale = "Urgent priority: Moderate physiological distress or severe non-immediate injury detected."
        else:
            suggested_priority = "GREEN"
            escalation_triggered = False
            rationale = "Delayed priority: Baseline physiological parameters are within acceptable bounds."

        return RuleEvaluationResult(
            activated_rules=activated,
            suggested_priority=suggested_priority,
            escalation_triggered=escalation_triggered,
            rationale=rationale
        )
