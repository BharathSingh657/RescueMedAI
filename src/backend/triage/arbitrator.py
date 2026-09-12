"""
Hybrid Triage Arbitrator for RescueMedAI.
Synthesizes explicit deterministic safety rules with probabilistic Bayesian evidence.
Guarantees conservative escalation when conflicting or high-risk signals are detected.

SAFETY NOTICE:
AI-Assisted Emergency Decision Support — Prototype — Human Clinical Review Required.
Outputs are decision-support classifications, not autonomous medical decisions.
"""

from typing import Dict, Any
from src.backend.state import (
    PatientRecord,
    TriageDecision,
    RuleEvaluationResult,
    BayesianEvaluationResult,
    MEDICAL_SAFETY_DISCLAIMER
)
from src.backend.triage.rules import EmergencyRuleEngine
from src.backend.triage.bayesian import EmergencyBayesianNetwork


class HybridTriageArbitrator:
    def __init__(self):
        self.rule_engine = EmergencyRuleEngine()
        self.bayesian_net = EmergencyBayesianNetwork()
        self.disclaimer = MEDICAL_SAFETY_DISCLAIMER

    def arbitrate(self, patient: PatientRecord) -> TriageDecision:
        """
        Executes both Rule Engine and Bayesian Network, then performs safe conservative arbitration.
        """
        rule_res: RuleEvaluationResult = self.rule_engine.evaluate(patient)
        bayes_res: BayesianEvaluationResult = self.bayesian_net.infer(patient)

        # Critical condition pool
        critical_conditions = [
            "Tension Pneumothorax / Respiratory Failure",
            "Hypovolemic / Hemorrhagic Shock",
            "Severe Traumatic Brain Injury",
            "Burn Shock / Inhalation Injury"
        ]
        sum_critical_prob = sum(
            bayes_res.hypotheses_posteriors.get(c, 0.0) for c in critical_conditions
        )
        top_critical_prob = max(
            bayes_res.hypotheses_posteriors.get(c, 0.0) for c in critical_conditions
        )

        conservative_escalation = False
        reasons = []

        # Decision Matrix with Conservative Escalation
        if rule_res.suggested_priority == "RED":
            final_priority = "RED"
            priority_level = "Immediate (Priority 1)"
            reasons.append("Explicit critical triage rule triggered.")
            if top_critical_prob > 0.40:
                reasons.append(f"Supported by Bayesian model: high likelihood of {bayes_res.top_condition} ({top_critical_prob:.1%}).")
        elif top_critical_prob >= 0.45 or sum_critical_prob >= 0.60:
            # Conservative escalation from Yellow/Green to Red due to strong probabilistic risk
            final_priority = "RED"
            priority_level = "Immediate (Priority 1)"
            conservative_escalation = True
            reasons.append(
                f"Conservative Escalation: While rule threshold was not breached, Bayesian inference detected "
                f"high cumulative critical risk ({sum_critical_prob:.1%}) favoring {bayes_res.top_condition}."
            )
        elif rule_res.suggested_priority == "YELLOW":
            final_priority = "YELLOW"
            priority_level = "Urgent (Priority 2)"
            reasons.append("Moderate physiological stress or significant injury patterns identified by rules.")
        elif sum_critical_prob >= 0.25:
            final_priority = "YELLOW"
            priority_level = "Urgent (Priority 2)"
            conservative_escalation = True
            reasons.append(
                f"Conservative Escalation: Borderline vitals elevated Bayesian critical risk to {sum_critical_prob:.1%}. Safe escalation to Yellow."
            )
        else:
            final_priority = "GREEN"
            priority_level = "Delayed (Priority 3)"
            reasons.append("All vital signs within acceptable bounds; low probability of critical pathology.")

        # Confidence metric: penalize high entropy (uncertainty)
        # Max entropy for 5 states is log2(5) ~= 2.32
        entropy_factor = max(0.0, 1.0 - (bayes_res.entropy / 2.32))
        rule_alignment = 1.0 if (final_priority == rule_res.suggested_priority) else 0.75
        confidence_score = round(min(0.98, max(0.40, (0.5 * rule_alignment) + (0.5 * entropy_factor))), 2)

        explanation = " ".join(reasons)

        return TriageDecision(
            priority=final_priority,
            priority_level=priority_level,
            rule_result=rule_res,
            bayesian_result=bayes_res,
            conservative_escalation_applied=conservative_escalation,
            explanation=explanation,
            confidence_score=confidence_score,
            disclaimer=self.disclaimer
        )
