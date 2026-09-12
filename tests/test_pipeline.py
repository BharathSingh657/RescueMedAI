"""
End-to-End Pipeline Integration Tests.
Verifies that all 3 scenarios execute completely through the single IncidentState contract,
confirming critical RED triage in Scenario 1, safest != shortest route in Scenario 2,
and stable GREEN triage in Scenario 3.
"""

import json
import pytest
from src.backend.pipeline import RescueMedPipeline
from src.backend.state import MEDICAL_SAFETY_DISCLAIMER


def load_test_scenarios():
    with open("data/sample_scenarios.json", "r") as f:
        return json.load(f)["scenarios"]


def test_scenario_1_earthquake_collapse_end_to_end():
    scenarios = load_test_scenarios()
    eq_scenario = next(s for s in scenarios if s["id"] == "SCENARIO-EQ-01")

    pipeline = RescueMedPipeline()
    state = pipeline.run_scenario(eq_scenario)

    # 1. State integrity
    assert state.disaster_type == "Earthquake"
    assert state.incident_node == "N-04"
    assert state.destination_node == "H-01"

    # 2. Vision & Severity
    assert state.severity is not None
    assert state.severity.label in ["High", "Critical"]
    assert state.severity.score >= 0.50

    # 3. Routing & Blockage Avoidance
    assert state.selected_route is not None
    assert state.selected_route.path
    assert state.selected_route.path[0] == "N-04"
    assert state.selected_route.path[-1] == "H-01"
    # Ensure blocked edge ("N-04", "N-05") is not in path
    for i in range(len(state.selected_route.path) - 1):
        edge = (state.selected_route.path[i], state.selected_route.path[i + 1])
        assert edge != ("N-04", "N-05")
        assert edge != ("N-05", "N-04")

    # 4. Medical Triage
    assert state.triage is not None
    assert state.triage.priority == "RED"
    assert state.triage.priority_level == "Immediate (Priority 1)"
    assert state.triage.disclaimer == MEDICAL_SAFETY_DISCLAIMER

    # 5. Resource Allocation
    assert state.resource_assignment is not None
    assert state.resource_assignment.ambulance_type == "ALS"
    assert "Trauma" in state.resource_assignment.hospital_trauma_level
    assert state.resource_assignment.rescue_team_type == "Heavy Extrication"

    # 6. Hospital Notification
    assert state.hospital_notification is not None
    assert state.hospital_notification.triage_color == "RED"
    assert "SBAR" in state.hospital_notification.sbar_formatted_text
    assert len(state.pipeline_log) >= 5


def test_scenario_2_flash_flood_safest_not_shortest():
    scenarios = load_test_scenarios()
    fl_scenario = next(s for s in scenarios if s["id"] == "SCENARIO-FL-02")

    pipeline = RescueMedPipeline()
    state = pipeline.run_scenario(fl_scenario)

    # 1. State integrity
    assert state.disaster_type == "Flood"
    assert state.incident_node == "N-08"
    assert state.destination_node == "H-02"

    # 2. Safest Route != Shortest Route Proof
    assert state.shortest_vs_safest is not None
    cmp = state.shortest_vs_safest
    assert cmp["is_safest_different_from_shortest"] is True
    # The safest route avoids the flooded N-08 -> N-09 arterial
    assert cmp["safest_route"].cost < cmp["shortest_route"].cost
    assert cmp["risk_cost_saved_min"] > 0

    # 3. Medical Triage
    assert state.triage is not None
    assert state.triage.priority == "YELLOW"
    assert state.triage.priority_level == "Urgent (Priority 2)"

    # 4. Resource Allocation
    assert state.resource_assignment is not None
    assert state.resource_assignment.rescue_team_type == "Water Rescue"


def test_scenario_3_minor_incident_stable():
    scenarios = load_test_scenarios()
    cl_scenario = next(s for s in scenarios if s["id"] == "SCENARIO-FR-03")

    pipeline = RescueMedPipeline()
    state = pipeline.run_scenario(cl_scenario)

    # 1. Severity
    assert state.severity is not None
    assert state.severity.label == "Low"

    # 2. Routing
    assert state.selected_route is not None
    assert len(state.selected_route.path) >= 2

    # 3. Medical Triage
    assert state.triage is not None
    assert state.triage.priority == "GREEN"
    assert state.triage.priority_level == "Delayed (Priority 3)"

    # 4. Resource Allocation Preserves Level 1
    assert state.resource_assignment is not None
    assert state.resource_assignment.hospital_trauma_level == "Community Care"
