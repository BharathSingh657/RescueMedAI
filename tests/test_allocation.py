"""
Unit tests for Resource Allocation module.
Verifies multi-factor scoring, baseline comparison, and simulation notices.
"""

from src.backend.routing.graph import RoadNetwork
from src.backend.allocation.allocator import EmergencyResourceAllocator
from src.backend.state import SIMULATED_DATA_NOTICE


def test_resource_allocation_red_priority():
    graph = RoadNetwork.create_default_city_grid()
    allocator = EmergencyResourceAllocator(graph)

    assignment = allocator.allocate_priority_aware(
        incident_node="N-04",
        triage_priority="RED",
        hazard_type="Structural Collapse"
    )

    assert assignment.ambulance_id is not None
    assert assignment.ambulance_type == "ALS"  # Red priority must prioritize ALS
    assert "Trauma" in assignment.hospital_trauma_level
    assert assignment.rescue_team_type == "Heavy Extrication"  # Collapse matches Extrication
    assert assignment.is_simulated is True


def test_resource_allocation_green_priority():
    graph = RoadNetwork.create_default_city_grid()
    allocator = EmergencyResourceAllocator(graph)

    assignment = allocator.allocate_priority_aware(
        incident_node="N-04",
        triage_priority="GREEN",
        hazard_type="Minor Flood"
    )

    assert assignment.ambulance_id is not None
    # Green priority should preserve Level 1 trauma beds, favoring Community Care
    assert assignment.hospital_trauma_level == "Community Care"


def test_compare_with_unprioritized_baseline():
    graph = RoadNetwork.create_default_city_grid()
    allocator = EmergencyResourceAllocator(graph)

    comp = allocator.compare_with_unprioritized_baseline("N-04", "RED")
    assert "priority_aware" in comp
    assert "unprioritized_baseline" in comp
    assert "metrics" in comp
    assert comp["notice"] == SIMULATED_DATA_NOTICE
