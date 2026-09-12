"""
Bayesian Reasoning Layer for Emergency Decision Support.
Implements an exact discrete Bayesian Network in pure NumPy to compute
posterior condition probabilities and probabilistic uncertainty (entropy)
under incomplete or noisy vital signs.

SAFETY NOTICE:
AI-Assisted Emergency Decision Support — Prototype — Human Clinical Review Required.
Probabilities represent educational decision-support likelihoods rather than clinical diagnoses.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import math
from src.backend.state import PatientRecord, BayesianEvaluationResult, MEDICAL_SAFETY_DISCLAIMER


class EmergencyBayesianNetwork:
    """
    Discrete Bayesian Network modeling causal relationships between emergency conditions
    and observable physiological indicators.
    """
    CONDITIONS = [
        "Tension Pneumothorax / Respiratory Failure",
        "Hypovolemic / Hemorrhagic Shock",
        "Severe Traumatic Brain Injury",
        "Burn Shock / Inhalation Injury",
        "Stable Trauma / Minor Injury"
    ]

    def __init__(self):
        self.conditions = self.CONDITIONS

        # Prior probabilities P(Condition) in a disaster scenario
        self.priors: Dict[str, float] = {
            "Tension Pneumothorax / Respiratory Failure": 0.12,
            "Hypovolemic / Hemorrhagic Shock": 0.15,
            "Severe Traumatic Brain Injury": 0.13,
            "Burn Shock / Inhalation Injury": 0.10,
            "Stable Trauma / Minor Injury": 0.50
        }

        # Conditional Probability Tables (CPTs): P(Evidence | Condition)
        # 1. SpO2: {Normal (>=94), Low (88-93), Critical (<88)}
        self.cpt_spo2: Dict[str, Dict[str, float]] = {
            "Tension Pneumothorax / Respiratory Failure": {"Normal": 0.05, "Low": 0.25, "Critical": 0.70},
            "Hypovolemic / Hemorrhagic Shock":           {"Normal": 0.30, "Low": 0.45, "Critical": 0.25},
            "Severe Traumatic Brain Injury":             {"Normal": 0.60, "Low": 0.30, "Critical": 0.10},
            "Burn Shock / Inhalation Injury":            {"Normal": 0.15, "Low": 0.35, "Critical": 0.50},
            "Stable Trauma / Minor Injury":              {"Normal": 0.90, "Low": 0.08, "Critical": 0.02}
        }

        # 2. Blood Pressure (Systolic): {Normal (>=100), Low (85-99), Critical (<85)}
        self.cpt_bp: Dict[str, Dict[str, float]] = {
            "Tension Pneumothorax / Respiratory Failure": {"Normal": 0.35, "Low": 0.40, "Critical": 0.25},
            "Hypovolemic / Hemorrhagic Shock":           {"Normal": 0.05, "Low": 0.25, "Critical": 0.70},
            "Severe Traumatic Brain Injury":             {"Normal": 0.55, "Low": 0.25, "Critical": 0.20},
            "Burn Shock / Inhalation Injury":            {"Normal": 0.30, "Low": 0.45, "Critical": 0.25},
            "Stable Trauma / Minor Injury":              {"Normal": 0.92, "Low": 0.06, "Critical": 0.02}
        }

        # 3. Heart Rate: {Normal (60-100), Tachycardia (101-130), Critical (>130 or <50)}
        self.cpt_hr: Dict[str, Dict[str, float]] = {
            "Tension Pneumothorax / Respiratory Failure": {"Normal": 0.15, "Tachycardia": 0.55, "Critical": 0.30},
            "Hypovolemic / Hemorrhagic Shock":           {"Normal": 0.05, "Tachycardia": 0.45, "Critical": 0.50},
            "Severe Traumatic Brain Injury":             {"Normal": 0.40, "Tachycardia": 0.30, "Critical": 0.30},
            "Burn Shock / Inhalation Injury":            {"Normal": 0.20, "Tachycardia": 0.55, "Critical": 0.25},
            "Stable Trauma / Minor Injury":              {"Normal": 0.85, "Tachycardia": 0.12, "Critical": 0.03}
        }

        # 4. Consciousness (AVPU): {Alert, Voice, Pain_or_Unresponsive}
        self.cpt_consciousness: Dict[str, Dict[str, float]] = {
            "Tension Pneumothorax / Respiratory Failure": {"Alert": 0.25, "Voice": 0.45, "Pain_or_Unresponsive": 0.30},
            "Hypovolemic / Hemorrhagic Shock":           {"Alert": 0.15, "Voice": 0.45, "Pain_or_Unresponsive": 0.40},
            "Severe Traumatic Brain Injury":             {"Alert": 0.05, "Voice": 0.25, "Pain_or_Unresponsive": 0.70},
            "Burn Shock / Inhalation Injury":            {"Alert": 0.40, "Voice": 0.40, "Pain_or_Unresponsive": 0.20},
            "Stable Trauma / Minor Injury":              {"Alert": 0.95, "Voice": 0.04, "Pain_or_Unresponsive": 0.01}
        }

        # 5. Symptom / Trauma finding: {Chest, Bleed, Head, Burns, Minor_or_None}
        self.cpt_symptom: Dict[str, Dict[str, float]] = {
            "Tension Pneumothorax / Respiratory Failure": {"Chest": 0.75, "Bleed": 0.05, "Head": 0.05, "Burns": 0.10, "Minor_or_None": 0.05},
            "Hypovolemic / Hemorrhagic Shock":           {"Chest": 0.10, "Bleed": 0.75, "Head": 0.05, "Burns": 0.05, "Minor_or_None": 0.05},
            "Severe Traumatic Brain Injury":             {"Chest": 0.05, "Bleed": 0.10, "Head": 0.80, "Burns": 0.02, "Minor_or_None": 0.03},
            "Burn Shock / Inhalation Injury":            {"Chest": 0.10, "Bleed": 0.02, "Head": 0.03, "Burns": 0.80, "Minor_or_None": 0.05},
            "Stable Trauma / Minor Injury":              {"Chest": 0.02, "Bleed": 0.03, "Head": 0.05, "Burns": 0.05, "Minor_or_None": 0.85}
        }

    def _discretize_vitals(self, patient: PatientRecord) -> Tuple[Dict[str, str], List[str]]:
        """Maps continuous patient vitals to discrete evidence states."""
        evidence = {}
        evidence_descriptions = []

        # SpO2
        if patient.spo2 >= 94:
            evidence["spo2"] = "Normal"
        elif patient.spo2 >= 88:
            evidence["spo2"] = "Low"
        else:
            evidence["spo2"] = "Critical"
        evidence_descriptions.append(f"SpO2: {patient.spo2}% ({evidence['spo2']})")

        # Systolic BP
        if patient.systolic_bp >= 100:
            evidence["bp"] = "Normal"
        elif patient.systolic_bp >= 85:
            evidence["bp"] = "Low"
        else:
            evidence["bp"] = "Critical"
        evidence_descriptions.append(f"Systolic BP: {patient.systolic_bp} mmHg ({evidence['bp']})")

        # Heart Rate
        if 60 <= patient.heart_rate <= 100:
            evidence["hr"] = "Normal"
        elif 101 <= patient.heart_rate <= 130:
            evidence["hr"] = "Tachycardia"
        else:
            evidence["hr"] = "Critical"
        evidence_descriptions.append(f"Heart Rate: {patient.heart_rate} bpm ({evidence['hr']})")

        # Consciousness
        if patient.consciousness == "Alert":
            evidence["consciousness"] = "Alert"
        elif patient.consciousness == "Voice":
            evidence["consciousness"] = "Voice"
        else:
            evidence["consciousness"] = "Pain_or_Unresponsive"
        evidence_descriptions.append(f"Consciousness: {patient.consciousness}")

        # Symptoms
        symptoms_str = " ".join(patient.symptoms).lower()
        if "chest" in symptoms_str or "breathing" in symptoms_str:
            evidence["symptom"] = "Chest"
        elif "bleed" in symptoms_str or "hemorrhage" in symptoms_str:
            evidence["symptom"] = "Bleed"
        elif "head" in symptoms_str or "brain" in symptoms_str or "concussion" in symptoms_str:
            evidence["symptom"] = "Head"
        elif "burn" in symptoms_str or "smoke" in symptoms_str:
            evidence["symptom"] = "Burns"
        else:
            evidence["symptom"] = "Minor_or_None"
        evidence_descriptions.append(f"Symptom Cluster: {evidence['symptom']}")

        return evidence, evidence_descriptions

    def infer(self, patient: PatientRecord) -> BayesianEvaluationResult:
        """
        Computes exact posterior probabilities P(Condition | Evidence) using Bayes' Theorem.
        """
        evidence, evidence_applied = self._discretize_vitals(patient)
        unnormalized = {}

        for c in self.conditions:
            prior = self.priors[c]
            # Likelihood product P(E1 | c) * P(E2 | c) * ...
            likelihood = (
                self.cpt_spo2[c][evidence["spo2"]] *
                self.cpt_bp[c][evidence["bp"]] *
                self.cpt_hr[c][evidence["hr"]] *
                self.cpt_consciousness[c][evidence["consciousness"]] *
                self.cpt_symptom[c][evidence["symptom"]]
            )
            unnormalized[c] = prior * likelihood

        total_evidence = sum(unnormalized.values())
        if total_evidence <= 0:
            # Fallback uniform if zero
            posteriors = {c: 1.0 / len(self.conditions) for c in self.conditions}
        else:
            posteriors = {c: float(val / total_evidence) for c, val in unnormalized.items()}

        # Identify highest probability condition
        top_condition = max(posteriors, key=posteriors.get)
        top_probability = posteriors[top_condition]

        # Calculate Shannon entropy: H = -sum(p * log2(p))
        entropy = 0.0
        for p in posteriors.values():
            if p > 1e-9:
                entropy -= p * math.log2(p)

        return BayesianEvaluationResult(
            hypotheses_posteriors={k: round(v, 4) for k, v in posteriors.items()},
            top_condition=top_condition,
            top_probability=round(top_probability, 4),
            entropy=round(entropy, 3),
            evidence_applied=evidence_applied
        )
