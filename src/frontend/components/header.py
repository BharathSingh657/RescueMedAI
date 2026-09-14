"""
Header and banner components for RescueMedAI Operations Command Center.
"""
import streamlit as st
import time

from src.backend.routing.search import a_star_search
from src.backend.hospital.notification import HospitalNotificationGenerator


def render_hero_header():
    """Renders the top tactical command center header banner."""
    st.markdown("""
    <div class="hero-banner">
        <div>
            <div class="hero-title">
                <span>🚑 RescueMedAI Command Center</span>
                <span class="hero-status-tag">● SYSTEM ONLINE</span>
                <span class="hero-alert-tag">ALERT LEVEL: ACTIVE DISASTER</span>
            </div>
            <div class="hero-subtitle">
                Autonomous Multi-Modal Disaster Support Platform • PyTorch Aerial Perception • Dynamic A* Rerouting • Priority Resource Dispatch • Hybrid Bayesian Field Triage
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_command_bar(apply_scenario_fn):
    """
    Renders the predefined scenario switcher toolbar.
    If a custom image is uploaded, this entire predefined bar is hidden/removed.
    """
    is_custom = bool(st.session_state.get("is_custom_image_uploaded", False) or (st.session_state.get("last_uploaded_file") is not None))

    # If custom image uploaded, remove the whole predefined scenario bar completely
    if is_custom:
        return

    if "scenarios" in st.session_state and st.session_state.scenarios:
        scenarios = st.session_state.scenarios
        c1, c2, c3, c4 = st.columns([1.5, 1.2, 1.4, 1.2])

        with c1:
            scenario_titles = [f"{s['title']} ({s['location_name']})" for s in scenarios]
            cur_idx = st.session_state.get("current_scenario_idx", 0)
            selected_idx = st.selectbox(
                "⚡ Select Preset Disaster Scenario:",
                range(len(scenarios)),
                format_func=lambda i: scenario_titles[i],
                index=cur_idx,
                key="top_scenario_selector"
            )
            if selected_idx != cur_idx:
                st.session_state.current_scenario_idx = selected_idx
                apply_scenario_fn(scenarios[selected_idx])
                st.rerun()

        state = st.session_state.state
        road_graph = st.session_state.graph

        with c2:
            st.metric("Disaster Type", state.disaster_type)

        with c3:
            nodes_list = list(road_graph.nodes.keys())
            if nodes_list:
                node_options = [
                    (nid, f"{nid} ({road_graph.nodes[nid]['name']})")
                    for nid in nodes_list
                ]
                cur_node_idx = nodes_list.index(state.incident_node) if state.incident_node in nodes_list else 0
                selected_node_tuple = st.selectbox(
                    "📍 Incident Location Node:",
                    node_options,
                    format_func=lambda x: x[1],
                    index=cur_node_idx,
                    key="top_incident_location_select"
                )
                selected_node_id = selected_node_tuple[0]
                if selected_node_id != state.incident_node:
                    state.incident_node = selected_node_id
                    state.location_name = road_graph.nodes[selected_node_id]["name"]
                    # Recalculate A* route & allocation
                    state.selected_route = a_star_search(road_graph, state.incident_node, state.destination_node)
                    triage_priority = state.triage.priority if state.triage else "RED"
                    state.resource_assignment = st.session_state.allocator.allocate_priority_aware(
                        state.incident_node, triage_priority, state.disaster_type
                    )
                    state.hospital_notification = HospitalNotificationGenerator.generate(state)
                    state.log_step("IncidentLocationChange", f"Changed incident location to {selected_node_id}")
                    st.rerun()
            else:
                st.metric("Incident Location", f"{state.incident_node} ({state.location_name})")

        with c4:
            t_prio = state.triage.priority if state.triage else "RED"
            st.metric("Triage Priority Level", t_prio)


def render_safety_notice(disclaimer: str):
    st.markdown(f"""
    <div class="safety-disclaimer-box">
        <span style="font-size: 1.2rem;">🛡️</span>
        <div><strong>CLINICAL SAFETY DIRECTIVE:</strong> {disclaimer}</div>
    </div>
    """, unsafe_allow_html=True)


def render_simulation_notice(notice: str):
    st.markdown(f"""
    <div class="simulation-notice-box">
        <div><strong>📍 PROTOTYPE / SIMULATED DATA NOTICE:</strong> {notice}</div>
    </div>
    """, unsafe_allow_html=True)
