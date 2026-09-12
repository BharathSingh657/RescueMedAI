"""
Resource Allocation Engine for RescueMedAI.
Assigns emergency response resources (Ambulances, Rescue Teams, Receiving Hospitals)
using multi-criteria priority scoring.

NOTICE:
Hospital capacity, ambulance status, and facility locations are simulated prototype data.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import copy
from src.backend.routing.graph import RoadNetwork
from src.backend.routing.search import a_star_search
from src.backend.state import ResourceAssignment, SIMULATED_DATA_NOTICE


@dataclass
class Ambulance:
    id: str
    unit_type: str            # 'ALS' or 'BLS'
    current_node: str
    is_available: bool = True
    speed_factor: float = 1.0


@dataclass
class RescueTeam:
    id: str
    specialty: str            # 'Heavy Extrication', 'Water Rescue', 'Fire Hazard', 'Medical Evacuation'
    current_node: str
    is_available: bool = True


@dataclass
class HospitalFacility:
    id: str
    name: str
    node: str
    trauma_level: str         # 'Level 1 Trauma', 'Level 2 Trauma', 'Community Care'
    er_beds_total: int
    er_beds_available: int
    icu_beds_total: int
    icu_beds_available: int
    has_burn_unit: bool = False
    has_neurosurgery: bool = False

    @property
    def capacity_ratio(self) -> float:
        if self.er_beds_total <= 0:
            return 0.0
        return self.er_beds_available / self.er_beds_total


class EmergencyResourceAllocator:
    def __init__(self, graph: RoadNetwork):
        self.graph = graph
        self.simulation_notice = SIMULATED_DATA_NOTICE
        self.ambulances: Dict[str, Ambulance] = self._create_default_fleet()
        self.rescue_teams: Dict[str, RescueTeam] = self._create_default_teams()
        self.hospitals: Dict[str, HospitalFacility] = self._create_default_hospitals()

    def _create_default_fleet(self) -> Dict[str, Ambulance]:
        return {
            "AMB-ALS-01": Ambulance("AMB-ALS-01", "ALS", "D-01", True),
            "AMB-ALS-02": Ambulance("AMB-ALS-02", "ALS", "N-01", True),
            "AMB-BLS-01": Ambulance("AMB-BLS-01", "BLS", "D-01", True),
            "AMB-BLS-02": Ambulance("AMB-BLS-02", "BLS", "N-08", True),
            "AMB-BLS-03": Ambulance("AMB-BLS-03", "BLS", "N-11", True)
        }

    def _create_default_teams(self) -> Dict[str, RescueTeam]:
        return {
            "RESCUE-01": RescueTeam("RESCUE-01", "Heavy Extrication", "D-01", True),
            "RESCUE-02": RescueTeam("RESCUE-02", "Water Rescue", "N-02", True),
            "RESCUE-03": RescueTeam("RESCUE-03", "Fire Hazard", "N-05", True),
            "RESCUE-04": RescueTeam("RESCUE-04", "Medical Evacuation", "D-01", True)
        }

    def _create_default_hospitals(self) -> Dict[str, HospitalFacility]:
        return {
            "H-01": HospitalFacility(
                id="H-01",
                name="Memorial Level-1 Trauma Center",
                node="H-01",
                trauma_level="Level 1 Trauma",
                er_beds_total=24,
                er_beds_available=6,
                icu_beds_total=12,
                icu_beds_available=3,
                has_burn_unit=True,
                has_neurosurgery=True
            ),
            "H-02": HospitalFacility(
                id="H-02",
                name="Metro General Hospital",
                node="H-02",
                trauma_level="Level 2 Trauma",
                er_beds_total=30,
                er_beds_available=14,
                icu_beds_total=10,
                icu_beds_available=5,
                has_burn_unit=False,
                has_neurosurgery=True
            ),
            "H-03": HospitalFacility(
                id="H-03",
                name="St. Jude Community Hospital",
                node="H-03",
                trauma_level="Community Care",
                er_beds_total=16,
                er_beds_available=9,
                icu_beds_total=4,
                icu_beds_available=2,
                has_burn_unit=False,
                has_neurosurgery=False
            )
        }

    def allocate_priority_aware(
        self,
        incident_node: str,
        triage_priority: str = "RED",
        hazard_type: str = "Structural Collapse"
    ) -> ResourceAssignment:
        """
        Priority-aware assignment balancing urgency, ambulance capabilities,
        road travel costs, and hospital trauma/ICU capacities.
        """
        urgency_weight = 3.0 if triage_priority == "RED" else (1.8 if triage_priority == "YELLOW" else 1.0)
        
        # 1. Select best ambulance
        best_amb: Optional[Ambulance] = None
        best_amb_eta = float('inf')
        best_amb_score = -float('inf')

        for amb in self.ambulances.values():
            if not amb.is_available:
                continue
            route = a_star_search(self.graph, amb.current_node, incident_node)
            travel_time = route.travel_time_min if route.path else 999.0
            
            # Score ambulance
            type_bonus = 1.5 if (triage_priority == "RED" and amb.unit_type == "ALS") else 1.0
            amb_score = (100.0 / (travel_time + 1.0)) * type_bonus

            if amb_score > best_amb_score:
                best_amb_score = amb_score
                best_amb = amb
                best_amb_eta = travel_time

        selected_amb = best_amb or list(self.ambulances.values())[0]

        # 2. Select appropriate rescue team
        best_team = None
        hz_lower = hazard_type.lower()
        for team in self.rescue_teams.values():
            if not team.is_available:
                continue
            if any(w in hz_lower for w in ["water", "flood", "submersion", "drowning"]) and team.specialty == "Water Rescue":
                best_team = team
                break
            elif any(w in hz_lower for w in ["fire", "burn", "wildfire", "smoke", "gas"]) and team.specialty == "Fire Hazard":
                best_team = team
                break
            elif any(w in hz_lower for w in ["collapse", "earthquake", "structural", "rubble", "trapped"]) and team.specialty == "Heavy Extrication":
                best_team = team
                break
            elif team.specialty == "Medical Evacuation":
                best_team = team
        selected_team = best_team or list(self.rescue_teams.values())[0]

        # 3. Select best hospital facility
        best_hosp: Optional[HospitalFacility] = None
        best_hosp_score = -float('inf')
        best_hosp_dist = 0.0

        for hosp in self.hospitals.values():
            route = a_star_search(self.graph, incident_node, hosp.node)
            dist_km = route.distance_km if route.path else 999.0
            travel_time = route.travel_time_min if route.path else 999.0

            # Priority-aware criteria
            capacity_score = hosp.capacity_ratio
            if triage_priority == "RED":
                trauma_bonus = 2.0 if hosp.trauma_level == "Level 1 Trauma" else (1.2 if hosp.trauma_level == "Level 2 Trauma" else 0.5)
                icu_score = (hosp.icu_beds_available / max(1, hosp.icu_beds_total))
            elif triage_priority == "YELLOW":
                trauma_bonus = 1.5 if hosp.trauma_level == "Level 2 Trauma" else 1.0
                icu_score = 0.5
            else: # GREEN
                # Preserve Level 1 for severe cases
                trauma_bonus = 2.0 if hosp.trauma_level == "Community Care" else 0.7
                icu_score = 0.2

            hosp_score = urgency_weight * (
                (trauma_bonus * 2.5) +
                (capacity_score * 2.0) +
                (icu_score * 1.5) +
                (10.0 / (travel_time + 1.0))
            )

            if hosp_score > best_hosp_score:
                best_hosp_score = hosp_score
                best_hosp = hosp
                best_hosp_dist = dist_km

        selected_hosp = best_hosp or list(self.hospitals.values())[0]

        total_allocation_score = round(best_amb_score + best_hosp_score, 2)

        return ResourceAssignment(
            ambulance_id=selected_amb.id,
            ambulance_type=selected_amb.unit_type,
            ambulance_eta_min=round(best_amb_eta, 1),
            rescue_team_id=selected_team.id,
            rescue_team_type=selected_team.specialty,
            hospital_id=selected_hosp.id,
            hospital_name=selected_hosp.name,
            hospital_trauma_level=selected_hosp.trauma_level,
            hospital_distance_km=round(best_hosp_dist, 1),
            allocation_score=total_allocation_score,
            is_simulated=True
        )

    def compare_with_unprioritized_baseline(
        self,
        incident_node: str,
        triage_priority: str = "RED"
    ) -> Dict[str, Any]:
        """
        Scenario C Benchmark: Compares priority-aware allocation against naive nearest-available baseline.
        """
        priority_assignment = self.allocate_priority_aware(incident_node, triage_priority)

        # Baseline: naive nearest available ambulance and nearest hospital (ignoring triage acuity & ICU capacity)
        nearest_amb = min(
            self.ambulances.values(),
            key=lambda a: a_star_search(self.graph, a.current_node, incident_node).travel_time_min
        )
        nearest_hosp = min(
            self.hospitals.values(),
            key=lambda h: a_star_search(self.graph, incident_node, h.node).distance_km
        )
        baseline_amb_eta = a_star_search(self.graph, nearest_amb.current_node, incident_node).travel_time_min
        baseline_hosp_dist = a_star_search(self.graph, incident_node, nearest_hosp.node).distance_km

        # Evaluate match quality
        priority_als_matched = (triage_priority == "RED" and priority_assignment.ambulance_type == "ALS")
        baseline_als_matched = (triage_priority == "RED" and nearest_amb.unit_type == "ALS")

        priority_trauma_matched = (triage_priority == "RED" and "Level 1" in priority_assignment.hospital_trauma_level)
        baseline_trauma_matched = (triage_priority == "RED" and "Level 1" in nearest_hosp.trauma_level)

        return {
            "priority_aware": {
                "ambulance": f"{priority_assignment.ambulance_id} ({priority_assignment.ambulance_type})",
                "hospital": f"{priority_assignment.hospital_name} ({priority_assignment.hospital_trauma_level})",
                "ambulance_eta_min": priority_assignment.ambulance_eta_min,
                "hospital_distance_km": priority_assignment.hospital_distance_km,
                "acuity_match_success": priority_als_matched and priority_trauma_matched
            },
            "unprioritized_baseline": {
                "ambulance": f"{nearest_amb.id} ({nearest_amb.unit_type})",
                "hospital": f"{nearest_hosp.name} ({nearest_hosp.trauma_level})",
                "ambulance_eta_min": round(baseline_amb_eta, 1),
                "hospital_distance_km": round(baseline_hosp_dist, 1),
                "acuity_match_success": baseline_als_matched and baseline_trauma_matched
            },
            "metrics": {
                "high_priority_coverage_gain": (1 if priority_als_matched else 0) - (1 if baseline_als_matched else 0),
                "trauma_capacity_preserved": (triage_priority == "GREEN" and "Community" in priority_assignment.hospital_trauma_level)
            },
            "notice": self.simulation_notice
        }
