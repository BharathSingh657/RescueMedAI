"""
RescueMedAI Unified Execution Pipeline.
Sequentially orchestrates all stages through the single IncidentState contract:
Perception -> Severity -> Dynamic Road Graph -> A* Safe Route -> Resource Allocation
-> Victim Vitals -> Rules & Bayesian Inference -> Triage -> Hospital Pre-Arrival Notification.
"""

from typing import Dict, Any, Optional
import os
import json
from PIL import Image

from src.backend.state import (
    IncidentState,
    PatientRecord,
    RouteResult,
    DetectionItem,
    SeverityAssessment,
    ResourceAssignment,
    TriageDecision,
    HospitalNotification
)
from src.backend.routing.graph import RoadNetwork
from src.backend.routing.search import a_star_search, compare_shortest_vs_safest
from src.backend.routing.benchmark import RoutingBenchmarkEngine
from src.backend.allocation.allocator import EmergencyResourceAllocator
from src.backend.vision.detector import DisasterVisionDetector
from src.backend.vision.severity import DisasterSeverityEstimator
from src.backend.triage.arbitrator import HybridTriageArbitrator
from src.backend.hospital.notification import HospitalNotificationGenerator


class RescueMedPipeline:
    """
    Unified end-to-end execution engine for RescueMedAI.
    Enforces that every stage reads and mutates the identical shared IncidentState instance.
    """

    def __init__(self, graph: Optional[RoadNetwork] = None):
        self.graph = graph or RoadNetwork.create_default_city_grid()
        self.allocator = EmergencyResourceAllocator(self.graph)
        self.arbitrator = HybridTriageArbitrator()

    def run_scenario(self, scenario_dict: Dict[str, Any]) -> IncidentState:
        """
        Executes a deterministic disaster scenario completely from end to end.
        """
        state = IncidentState()
        state.disaster_type = scenario_dict["disaster_type"]
        state.location_name = scenario_dict["location_name"]
        state.incident_node = scenario_dict["incident_node"]
        state.destination_node = scenario_dict["target_hospital"]

        # 1. Reset and configure Road Graph
        self.graph = RoadNetwork.create_default_city_grid()
        for u, v in scenario_dict.get("blocked_edges", []):
            self.graph.set_edge_status(u, v, is_blocked=True)
            state.blocked_edges.append((u, v))

        for edge_str, weight in scenario_dict.get("hazard_edges", {}).items():
            u, v = edge_str.split(",")
            self.graph.set_edge_status(u, v, is_blocked=False, hazard_weight=float(weight))
            state.hazard_edges[(u, v)] = float(weight)

        self.allocator = EmergencyResourceAllocator(self.graph)

        # 2. Stage 1 & 2: Computer Vision & Severity Estimation
        img_name = scenario_dict.get("image_file", "earthquake_collapse.jpg")
        img_path = os.path.join("data/sample_images", img_name)
        state.image_name = img_name
        state.image_path = img_path

        if os.path.exists(img_path):
            pil_img = Image.open(img_path)
            benchmark_anns = scenario_dict.get("benchmark_vision_annotations")
            detections = DisasterVisionDetector.detect_hazards(pil_img, benchmark_annotations=benchmark_anns)
            state.detections = detections
            state.severity = DisasterSeverityEstimator.estimate(detections)
            state.log_step("Vision&Severity", f"Assessed severity: {state.severity.label} ({state.severity.score})")
        else:
            state.severity = SeverityAssessment(score=0.5, label="Medium", rationale="Default fallback")

        # 3. Stage 3: Routing & Heuristic Search
        # Dynamic A* safe route avoiding hazard & blockages
        state.selected_route = a_star_search(self.graph, state.incident_node, state.destination_node)
        # Search benchmark across all 4 algorithms on identical graph
        bench_out = RoutingBenchmarkEngine.run_benchmark(self.graph, state.incident_node, state.destination_node)
        state.route_benchmarks = bench_out["results"]
        # Safest vs Shortest Distance comparison
        state.shortest_vs_safest = compare_shortest_vs_safest(self.graph, state.incident_node, state.destination_node)
        state.log_step(
            "Routing",
            f"A* path found: {' -> '.join(state.selected_route.path)} (cost: {state.selected_route.cost:.1f} min)",
            {"nodes_expanded": state.selected_route.nodes_expanded, "distance_km": state.selected_route.distance_km}
        )

        # 4. Stage 4 & 5: Victim Ingestion & Hybrid Medical Triage
        p_data = scenario_dict.get("default_patient")
        if p_data:
            state.patient = PatientRecord(**p_data)
            state.triage = self.arbitrator.arbitrate(state.patient)
            state.log_step(
                "MedicalTriage",
                f"Triage assigned: {state.triage.priority} ({state.triage.priority_level})",
                {"top_condition": state.triage.bayesian_result.top_condition, "confidence": state.triage.confidence_score}
            )

        # 5. Stage 6: Resource Allocation
        triage_prio = state.triage.priority if state.triage else "RED"
        state.resource_assignment = self.allocator.allocate_priority_aware(
            state.incident_node, triage_prio, state.disaster_type
        )
        state.log_step(
            "ResourceAllocation",
            f"Assigned {state.resource_assignment.ambulance_id} ({state.resource_assignment.ambulance_type}) to {state.resource_assignment.hospital_name}"
        )

        # 6. Stage 7: Hospital Pre-Arrival SBAR Notification
        state.hospital_notification = HospitalNotificationGenerator.generate(state)
        state.log_step("HospitalNotification", f"Dispatched SBAR report for {state.hospital_notification.case_id}")

        return state

    def recompute_from_patient(self, state: IncidentState, patient: PatientRecord) -> IncidentState:
        """
        Recomputes triage, resource allocation, and hospital dispatch when responder modifies patient vitals.
        """
        state.patient = patient
        state.triage = self.arbitrator.arbitrate(patient)
        triage_prio = state.triage.priority
        state.resource_assignment = self.allocator.allocate_priority_aware(
            state.incident_node, triage_prio, state.disaster_type
        )
        state.hospital_notification = HospitalNotificationGenerator.generate(state)
        state.log_step("ResponderUpdate", f"Updated vitals: Triage escalated to {triage_prio}")
        return state
