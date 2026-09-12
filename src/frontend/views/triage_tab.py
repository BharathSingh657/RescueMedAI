"""
Field Medical Triage View (Tab 4).
Interactive responder vitals input form and hybrid AI decision support visualization
(Rules + Bayesian probability distribution).
"""
import streamlit as st
import pandas as pd

from src.backend.state import PatientRecord, MEDICAL_SAFETY_DISCLAIMER
from src.backend.hospital.notification import HospitalNotificationGenerator
from src.frontend.components.header import render_safety_notice


def render_triage_tab():
    state = st.session_state.state

    st.subheader("Stage 5 & 6: Medical Decision Support Layer (Rules + Bayesian Inference)")
    render_safety_notice(MEDICAL_SAFETY_DISCLAIMER)

    tcol1, tcol2 = st.columns([1.1, 1.2])

    with tcol1:
        st.markdown("##### 📝 Responder Patient Vitals Ingestion")
        pt = state.patient or PatientRecord(
            patient_id="PT-DEFAULT", age=30, systolic_bp=110, diastolic_bp=70,
            spo2=96, heart_rate=80, respiratory_rate=18, consciousness="Alert"
        )

        with st.form("patient_vitals_form"):
            f_age = st.number_input("Patient Age:", min_value=1, max_value=110, value=int(pt.age))
            
            c_bp1, c_bp2 = st.columns(2)
            f_sys = c_bp1.number_input("Systolic BP (mmHg):", min_value=40, max_value=240, value=int(pt.systolic_bp))
            f_dia = c_bp2.number_input("Diastolic BP (mmHg):", min_value=20, max_value=140, value=int(pt.diastolic_bp))

            c_v1, c_v2 = st.columns(2)
            f_spo2 = c_v1.slider("SpO2 Oxygen Saturation (%):", min_value=50, max_value=100, value=int(pt.spo2))
            f_hr = c_v2.number_input("Heart Rate (bpm):", min_value=25, max_value=220, value=int(pt.heart_rate))

            c_v3, c_v4 = st.columns(2)
            f_rr = c_v3.number_input("Respiratory Rate (/min):", min_value=4, max_value=60, value=int(pt.respiratory_rate))
            f_avpu = c_v4.selectbox("Consciousness (AVPU):", ["Alert", "Voice", "Pain", "Unresponsive"], index=["Alert", "Voice", "Pain", "Unresponsive"].index(pt.consciousness) if pt.consciousness in ["Alert", "Voice", "Pain", "Unresponsive"] else 0)

            f_amb = st.checkbox("Ambulatory (Able to walk independently)", value=bool(pt.ambulatory))

            all_symptoms = [
                "Massive External Bleeding",
                "Penetrating Chest Trauma",
                "Suspected Limb Fracture",
                "Severe Smoke Inhalation",
                "Moderate Burns (<20%)",
                "Minor arm abrasion",
                "Difficulty breathing",
                "Mild hypothermia"
            ]
            default_selected = [s for s in pt.symptoms if s in all_symptoms]
            f_symptoms = st.multiselect("Observed Trauma Findings / Symptoms:", all_symptoms, default=default_selected)

            submitted = st.form_submit_button("⚡ Recompute Decision Support Triage", use_container_width=True)
            if submitted:
                updated_patient = PatientRecord(
                    patient_id=pt.patient_id,
                    age=f_age,
                    systolic_bp=f_sys,
                    diastolic_bp=f_dia,
                    spo2=f_spo2,
                    heart_rate=f_hr,
                    respiratory_rate=f_rr,
                    consciousness=f_avpu,
                    symptoms=f_symptoms,
                    visible_trauma=pt.visible_trauma,
                    ambulatory=f_amb
                )
                state.patient = updated_patient
                state.triage = st.session_state.arbitrator.arbitrate(updated_patient)
                # Update downstream allocation & notification
                state.resource_assignment = st.session_state.allocator.allocate_priority_aware(
                    state.incident_node, state.triage.priority, state.disaster_type
                )
                state.hospital_notification = HospitalNotificationGenerator.generate(state)
                st.success("Triage decision and hospital dispatch updated.")
                st.rerun()

    with tcol2:
        st.markdown("##### 🎯 AI Triage Decision Support Synthesis")
        if state.triage:
            tr = state.triage
            badge_class = f"triage-badge-{tr.priority.lower()}"
            st.markdown(f"""
            <div style="margin-bottom: 14px;">
                <span class="{badge_class}">{tr.priority} — {tr.priority_level}</span>
                <span style="color: #94a3b8; font-size: 0.9rem; margin-left: 12px;">Confidence Metric: <strong>{tr.confidence_score:.0%}</strong></span>
            </div>
            """, unsafe_allow_html=True)

            if tr.conservative_escalation_applied:
                st.warning(f"🛡️ **Conservative Safety Escalation Applied:** {tr.explanation}")
            else:
                st.info(f"💡 **Clinical Decision Rationale:** {tr.explanation}")

            # Bayesian Probabilities
            st.markdown("##### 🎲 Bayesian Pathological Likelihood Distribution")
            st.caption("Discrete Conditional Probability Table (CPT) inference under uncertain physiological signs:")

            bayes = tr.bayesian_result
            probs_df = pd.DataFrame([
                {"Candidate Pathology": cond, "Posterior Probability": prob}
                for cond, prob in sorted(bayes.hypotheses_posteriors.items(), key=lambda x: x[1], reverse=True)
            ])
            st.bar_chart(probs_df.set_index("Candidate Pathology"), height=220)

            st.caption(f"Shannon Entropy (Probabilistic Uncertainty): `{bayes.entropy:.3f} bits` | Top Indicated Condition: **{bayes.top_condition} ({bayes.top_probability:.1%})**")

            # Rule Engine activations
            st.markdown("##### 📜 Activated Deterministic Safety Rules")
            if tr.rule_result.activated_rules:
                for r in tr.rule_result.activated_rules:
                    p_color = "#ef4444" if r['priority'] == 'RED' else ("#f59e0b" if r['priority'] == 'YELLOW' else "#10b981")
                    st.markdown(f"""
                    <div style="border-left: 4px solid {p_color}; background: #131c2e; padding: 8px 12px; margin-bottom: 6px; border-radius: 4px;">
                        <strong style="color: {p_color};">[{r['rule_id']}] {r['name']}</strong><br/>
                        <span style="font-size: 0.82rem; color: #cbd5e1;">Condition: {r['condition']}</span><br/>
                        <span style="font-size: 0.82rem; color: #94a3b8;">Rationale: {r['rationale']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("No emergency rule thresholds triggered.")
