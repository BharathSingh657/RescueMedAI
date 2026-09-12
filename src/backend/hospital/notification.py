"""
Hospital Pre-Arrival Notification Module.
Generates structured emergency handoff transmissions (SBAR / MIST protocol)
for receiving trauma centers.

SAFETY NOTICE:
AI-Assisted Emergency Decision Support — Prototype — Human Clinical Review Required.
Hospital capacity, ambulance telemetry, GPS locations, and facility communications
are simulated for offline prototype demonstration.
"""

from typing import Dict, Any, List
import time
from src.backend.state import (
    IncidentState,
    HospitalNotification,
    MEDICAL_SAFETY_DISCLAIMER,
    SIMULATED_DATA_NOTICE
)


class HospitalNotificationGenerator:
    """
    Constructs pre-arrival clinical summaries adhering to standardized emergency handoff frameworks.
    """

    @staticmethod
    def generate(state: IncidentState) -> HospitalNotification:
        patient = state.patient
        triage = state.triage
        allocation = state.resource_assignment
        route = state.selected_route

        # Case identifiers
        case_id = f"CASE-{state.incident_id[-6:]}"
        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        # Vitals summary
        if patient:
            vitals_dict = {
                "age": patient.age,
                "bp": f"{patient.systolic_bp}/{patient.diastolic_bp} mmHg",
                "spo2": f"{patient.spo2}%",
                "heart_rate": f"{patient.heart_rate} bpm",
                "respiratory_rate": f"{patient.respiratory_rate}/min",
                "consciousness": patient.consciousness,
                "symptoms": patient.symptoms,
                "visible_trauma": patient.visible_trauma
            }
        else:
            vitals_dict = {"status": "Vitals not yet recorded"}

        # Bayesian suspected pathologies
        suspected = []
        if triage and triage.bayesian_result:
            for cond, prob in sorted(
                triage.bayesian_result.hypotheses_posteriors.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if prob >= 0.10:
                    suspected.append({"condition": cond, "posterior_probability": prob})

        # Transport telemetry
        transport = {
            "unit": allocation.ambulance_id if allocation else "AMB-ALS-01",
            "type": allocation.ambulance_type if allocation else "ALS",
            "eta_min": allocation.ambulance_eta_min if allocation else 12.0,
            "route_cost": round(route.cost, 1) if route else 15.0,
            "distance_km": round(route.distance_km, 1) if route else 6.5,
            "rescue_team": allocation.rescue_team_id if allocation else "RESCUE-01"
        }

        hosp_id = allocation.hospital_id if allocation else "H-01"
        hosp_name = allocation.hospital_name if allocation else "Memorial Level-1 Trauma Center"

        # Preparedness suggestions based on triage priority
        triage_color = triage.priority if triage else "RED"
        prep_items = []
        if triage_color == "RED":
            prep_items.extend([
                "Activate Trauma Resuscitation Bay 1 (High Acuity)",
                "On-call Trauma Surgeon and Anesthesia alert",
                "Prepare Rapid Sequence Intubation (RSI) equipment",
                "Standby 2 Units uncrossed O-Negative blood",
                "Prepare portable ultrasound (FAST exam)"
            ])
        elif triage_color == "YELLOW":
            prep_items.extend([
                "Assign Urgent Evaluation Cubicle",
                "Secondary survey and orthopedic evaluation standby",
                "Peripheral IV line and continuous monitoring setup"
            ])
        else:
            prep_items.extend([
                "Assign Delayed Observation Area",
                "Wound care and basic nursing assessment on arrival"
            ])

        # SBAR Text Formatter
        sbar = f"""### EMERGENCY PRE-ARRIVAL DISPATCH REPORT (SBAR)
**Case Reference:** {case_id} | **Incident:** {state.incident_id} | **Timestamp:** {now_str}
**Destination:** {hosp_name} ({hosp_id})

---
#### 1. SITUATION
- **Triage Acuity:** **{triage_color} (Priority Level: {triage.priority_level if triage else 'Immediate'})**
- **Disaster Context:** {state.disaster_type} at {state.location_name} (Estimated Disaster Severity: {state.severity.label if state.severity else 'High'})
- **Inbound Unit:** {transport['unit']} ({transport['type']}) | **ETA to Receiving Bay:** ~{transport['eta_min']} min

#### 2. BACKGROUND & MECHANISM
- **Patient Profile:** {vitals_dict.get('age', 'Unknown')}-year-old individual involved in {state.disaster_type}.
- **Reported Trauma / Symptoms:** {', '.join(vitals_dict.get('symptoms', [])) if vitals_dict.get('symptoms') else 'None recorded'}.
- **Visible Findings:** {vitals_dict.get('visible_trauma', 'None')}.

#### 3. ASSESSMENT (AI-Assisted Decision Support)
- **Primary Vital Signs:** BP {vitals_dict.get('bp', 'N/A')} | SpO2 {vitals_dict.get('spo2', 'N/A')} | HR {vitals_dict.get('heart_rate', 'N/A')} | RR {vitals_dict.get('respiratory_rate', 'N/A')} | Mental Status: {vitals_dict.get('consciousness', 'N/A')}
- **Rule Engine Findings:** {triage.rule_result.rationale if triage and triage.rule_result else 'Pending'}
- **Bayesian Risk Distribution:**
"""
        for item in suspected[:3]:
            sbar += f"  - *{item['condition']}*: {item['posterior_probability']:.1%}\n"

        sbar += f"""
#### 4. RECOMMENDATION & FACILITY PREPAREDNESS
"""
        for p in prep_items:
            sbar += f"- {p}\n"

        sbar += f"""
---
> **SAFETY NOTICE:** {MEDICAL_SAFETY_DISCLAIMER}
> **PROTOTYPE NOTICE:** {SIMULATED_DATA_NOTICE}
"""

        return HospitalNotification(
            case_id=case_id,
            timestamp=now_str,
            triage_color=triage_color,
            patient_vitals_summary=vitals_dict,
            suspected_pathologies=suspected,
            assigned_transport=transport,
            hospital_id=hosp_id,
            hospital_name=hosp_name,
            sbar_formatted_text=sbar.strip(),
            recommended_preparedness=prep_items,
            disclaimer=MEDICAL_SAFETY_DISCLAIMER,
            simulation_notice=SIMULATED_DATA_NOTICE
        )
