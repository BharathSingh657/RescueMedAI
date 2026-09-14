"""
Emergency Resource Allocation View (Tab 3).
Displays priority-aware ambulance, rescue crew, and hospital assignment matching versus baseline.
"""
import streamlit as st
import pandas as pd

from src.backend.allocation.allocator import EmergencyResourceAllocator
from src.backend.state import SIMULATED_DATA_NOTICE
from src.frontend.components.header import render_simulation_notice


def render_allocation_tab():
    state = st.session_state.state

    st.subheader("🚑 Emergency Resource Allocation & Multi-Criteria Matching")
    st.caption("Matches incident severity and field patient triage priority to optimal ambulance units, specialized rescue teams, and trauma hospital capacity.")

    render_simulation_notice(SIMULATED_DATA_NOTICE)

    allocator: EmergencyResourceAllocator = st.session_state.allocator
    assignment = state.resource_assignment

    acol1, acol2 = st.columns([1.1, 1.0])

    with acol1:
        st.markdown("##### 🚑 Dispatched Emergency Units")
        if assignment:
            st.markdown(f"""
            <div class="ops-card">
                <div class="ops-card-title">DISPATCHED AMBULANCE UNIT</div>
                <div class="ops-card-value">{assignment.ambulance_id} ({assignment.ambulance_type})</div>
                <div class="ops-card-sub">Estimated Dispatch ETA to Incident: <strong>{assignment.ambulance_eta_min:.1f} min</strong></div>
            </div>
            <div class="ops-card">
                <div class="ops-card-title">DISPATCHED SPECIALIST RESCUE CREW</div>
                <div class="ops-card-value">{assignment.rescue_team_id}</div>
                <div class="ops-card-sub">Specialty: <strong>{assignment.rescue_team_type}</strong></div>
            </div>
            <div class="ops-card">
                <div class="ops-card-title">DESTINATION RECEIVING TRAUMA FACILITY</div>
                <div class="ops-card-value">{assignment.hospital_name}</div>
                <div class="ops-card-sub">Designation: <strong>{assignment.hospital_trauma_level}</strong> | Road Distance: <strong>{assignment.hospital_distance_km:.1f} km</strong></div>
            </div>
            """, unsafe_allow_html=True)

    with acol2:
        st.markdown("##### ⚖️ Priority-Aware vs Baseline Allocation Comparison")
        st.caption("Demonstrating acuity-based hospital capacity preservation:")

        comp = allocator.compare_with_unprioritized_baseline(
            state.incident_node,
            state.triage.priority if state.triage else "RED"
        )
        
        comp_df = pd.DataFrame([
            {
                "Strategy": "Priority-Aware Allocation (Ours)",
                "Ambulance Assigned": comp["priority_aware"]["ambulance"],
                "Hospital Target": comp["priority_aware"]["hospital"],
                "Ambulance ETA": f"{comp['priority_aware']['ambulance_eta_min']} min",
                "Acuity Match": "Optimal Match" if comp["priority_aware"]["acuity_match_success"] else "Adequate"
            },
            {
                "Strategy": "Naive Nearest Baseline",
                "Ambulance Assigned": comp["unprioritized_baseline"]["ambulance"],
                "Hospital Target": comp["unprioritized_baseline"]["hospital"],
                "Ambulance ETA": f"{comp['unprioritized_baseline']['ambulance_eta_min']} min",
                "Acuity Match": "Suboptimal / Overloaded" if not comp["unprioritized_baseline"]["acuity_match_success"] else "Match"
            }
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        st.markdown("##### 🏥 Regional Trauma Center Capacity (Simulated Live Feed)")
        hosp_data = []
        for h in allocator.hospitals.values():
            hosp_data.append({
                "Hospital Facility": h.name,
                "Trauma Level": h.trauma_level,
                "ER Beds Available": f"{h.er_beds_available} / {h.er_beds_total}",
                "ICU Beds Available": f"{h.icu_beds_available} / {h.icu_beds_total}",
                "Burn Unit": "Yes" if h.has_burn_unit else "No"
            })
        st.dataframe(pd.DataFrame(hosp_data), hide_index=True, use_container_width=True)
