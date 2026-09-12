"""
RescueMedAI - Main Frontend App Launcher.
Orchestrates Streamlit session state, sidebar command console, and tab view routing.
"""
import streamlit as st
import json
import os
from PIL import Image

from src.backend.state import (
    IncidentState,
    PatientRecord,
    MEDICAL_SAFETY_DISCLAIMER,
)
from src.backend.routing.graph import RoadNetwork
from src.backend.routing.search import a_star_search
from src.backend.allocation.allocator import EmergencyResourceAllocator
from src.backend.vision.detector import DisasterVisionDetector
from src.backend.vision.severity import DisasterSeverityEstimator
from src.backend.triage.arbitrator import HybridTriageArbitrator
from src.backend.hospital.notification import HospitalNotificationGenerator

from src.frontend.components.styles import apply_custom_css
from src.frontend.components.header import render_hero_header, render_safety_notice
from src.frontend.views.vision_tab import render_vision_tab
from src.frontend.views.routing_tab import render_routing_tab
from src.frontend.views.allocation_tab import render_allocation_tab
from src.frontend.views.triage_tab import render_triage_tab
from src.frontend.views.notification_tab import render_notification_tab
from src.frontend.views.benchmark_tab import render_benchmark_tab


def load_scenarios():
    path = "data/sample_scenarios.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f).get("scenarios", [])
    return []


def init_incident_state():
    scenarios = load_scenarios()
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = scenarios

    if "current_scenario_idx" not in st.session_state:
        st.session_state.current_scenario_idx = 0

    if "state" not in st.session_state:
        st.session_state.state = IncidentState()

    if "graph" not in st.session_state:
        st.session_state.graph = RoadNetwork.create_default_city_grid()

    if "allocator" not in st.session_state:
        st.session_state.allocator = EmergencyResourceAllocator(st.session_state.graph)

    if "arbitrator" not in st.session_state:
        st.session_state.arbitrator = HybridTriageArbitrator()


def apply_scenario(scenario_dict):
    """Loads a deterministic disaster scenario into the single incident state."""
    state = st.session_state.state
    state.disaster_type = scenario_dict["disaster_type"]
    state.location_name = scenario_dict["location_name"]
    state.incident_node = scenario_dict["incident_node"]
    state.destination_node = scenario_dict["target_hospital"]

    # Reset and configure graph
    graph = RoadNetwork.create_default_city_grid()
    for u, v in scenario_dict.get("blocked_edges", []):
        graph.set_edge_status(u, v, is_blocked=True)
    for edge_str, weight in scenario_dict.get("hazard_edges", {}).items():
        u, v = edge_str.split(",")
        graph.set_edge_status(u, v, is_blocked=False, hazard_weight=float(weight))

    st.session_state.graph = graph
    st.session_state.allocator = EmergencyResourceAllocator(graph)

    # Ingest image & vision detections
    img_path = os.path.join("data/sample_images", scenario_dict.get("image_file", "earthquake_collapse.jpg"))
    state.image_path = img_path
    state.image_name = scenario_dict.get("image_file", "image.jpg")

    if os.path.exists(img_path):
        pil_img = Image.open(img_path)
        benchmark_anns = scenario_dict.get("benchmark_vision_annotations")
        detections = DisasterVisionDetector.detect_hazards(pil_img, benchmark_annotations=benchmark_anns)
        state.detections = detections
        state.severity = DisasterSeverityEstimator.estimate(detections)

    # Ingest default patient
    p_data = scenario_dict.get("default_patient")
    if p_data:
        state.patient = PatientRecord(**p_data)
        state.triage = st.session_state.arbitrator.arbitrate(state.patient)

    # Execute Routing & Allocation
    state.selected_route = a_star_search(graph, state.incident_node, state.destination_node)
    triage_priority = state.triage.priority if state.triage else "RED"
    state.resource_assignment = st.session_state.allocator.allocate_priority_aware(
        state.incident_node, triage_priority, state.disaster_type
    )

    # Generate Hospital Pre-Arrival Notification
    state.hospital_notification = HospitalNotificationGenerator.generate(state)
    st.session_state["last_uploaded_file"] = None
    state.log_step("ScenarioLoader", f"Applied scenario: {scenario_dict['title']}")


def main():
    apply_custom_css()
    init_incident_state()

    # Check if scenario needs to be loaded initially
    if "scenario_initialized" not in st.session_state and st.session_state.scenarios:
        apply_scenario(st.session_state.scenarios[0])
        st.session_state.scenario_initialized = True

    # SIDEBAR: Operational Controls
    with st.sidebar:
        st.markdown("### 🎛️ Command Console")
        scenario_titles = [s["title"] for s in st.session_state.scenarios]
        selected_scenario_idx = st.selectbox(
            "Select Disaster Incident Scenario:",
            range(len(scenario_titles)),
            format_func=lambda i: scenario_titles[i],
            index=st.session_state.current_scenario_idx
        )

        if selected_scenario_idx != st.session_state.current_scenario_idx:
            st.session_state.current_scenario_idx = selected_scenario_idx
            apply_scenario(st.session_state.scenarios[selected_scenario_idx])
            st.rerun()

        if st.button("🔄 Reset / Reload Scenario State", use_container_width=True):
            apply_scenario(st.session_state.scenarios[st.session_state.current_scenario_idx])
            st.success("Incident State reloaded successfully.")
            st.rerun()

        st.markdown("---")
        st.markdown("#### 📋 Incident Telemetry")
        state = st.session_state.state

        st.markdown(f"**Incident ID:** `{state.incident_id}`")
        st.markdown(f"**Disaster Event:** {state.disaster_type}")
        st.markdown(f"**Location:** {state.location_name} (`{state.incident_node}`)")
        st.markdown(f"**Severity Level:** **{state.severity.label if state.severity else 'N/A'}**")

        # Current Triage Badge
        if state.triage:
            p = state.triage.priority
            badge_class = f"triage-badge-{p.lower()}"
            st.markdown(f"""
            <div style="margin: 8px 0;">
                <span class="{badge_class}">{p} — {state.triage.priority_level}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"**Target Hospital:** `{state.destination_node}`")
        if state.resource_assignment:
            st.markdown(f"**Assigned Transport:** `{state.resource_assignment.ambulance_id} ({state.resource_assignment.ambulance_type})`")

        st.markdown("---")
        st.caption("RescueMedAI • Autonomous & Decision Support Systems Project")

    # MAIN INTERFACE
    render_hero_header()
    render_safety_notice(MEDICAL_SAFETY_DISCLAIMER)

    tabs = st.tabs([
        "🛰️ 1. Perception & Severity",
        "🗺️ 2. Road Graph & A* Search",
        "🚑 3. Resource Allocation",
        "🩺 4. Field Medical Triage",
        "🏥 5. Hospital Pre-Arrival Console",
        "📊 6. Academic Review & Benchmarks"
    ])

    with tabs[0]:
        render_vision_tab()

    with tabs[1]:
        render_routing_tab()

    with tabs[2]:
        render_allocation_tab()

    with tabs[3]:
        render_triage_tab()

    with tabs[4]:
        render_notification_tab()

    with tabs[5]:
        render_benchmark_tab()


if __name__ == "__main__":
    main()
