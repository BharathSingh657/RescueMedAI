"""
RescueMedAI - Incident State Architecture
Defines the unified state contract passed through all 9 stages of the response pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import uuid

# Mandatory safety disclaimer across all decision-support outputs
MEDICAL_SAFETY_DISCLAIMER = (
    "AI-Assisted Emergency Decision Support — Prototype — Human Clinical Review Required. "
    "This system provides educational decision support and does not provide clinical diagnosis, "
    "autonomous medical triage, or validated therapeutic directives."
)

SIMULATED_DATA_NOTICE = (
    "Hospital capacity, ambulance telemetry, GPS locations, and facility communications "
    "are simulated for offline prototype demonstration."
)


@dataclass
class DetectionItem:
    label: str                     # e.g., 'damaged_building', 'fire_indicator', 'blocked_road'
    confidence: float              # 0.0 - 1.0
    bounding_box: Optional[List[int]] = None  # [x1, y1, x2, y2]
    source: str = "local_vision_analysis"     # or "demo_scenario_dataset"
    details: str = ""


@dataclass
class RoadConditionAssessment:
    segment_id: str                      # e.g., "Corridor #1 (Primary Arterial)"
    condition: str                       # 'Normal', 'Partially Damaged', 'Severely Damaged', 'Flooded', 'Blocked'
    confidence: float                    # 0.0 - 1.0
    flood_coverage_pct: float = 0.0      # 0 to 100%
    debris_blockage_pct: float = 0.0     # 0 to 100%
    structural_damage_pct: float = 0.0   # 0 to 100%
    bounding_box: Optional[List[int]] = None # [x1, y1, x2, y2]
    suggested_hazard_weight: float = 0.0 # 0.0 = clear, 2.0 = moderate, 5.0 = severe
    is_impassable: bool = False
    details: str = ""


@dataclass
class SeverityAssessment:
    score: float                   # 0.0 to 1.0
    label: str                     # 'Low', 'Medium', 'High', 'Critical'
    damaged_building_ratio: float = 0.0
    blocked_road_count: int = 0
    fire_detected: bool = False
    affected_zone_density: float = 0.0
    rationale: str = ""
    road_condition_summary: str = ""


@dataclass
class RouteResult:
    algorithm: str
    path: List[str]
    cost: float
    distance_km: float
    travel_time_min: float
    nodes_expanded: int
    runtime_ms: float
    is_optimal: bool = True
    blocked_roads_avoided: int = 0


@dataclass
class ResourceAssignment:
    ambulance_id: str
    ambulance_type: str            # 'ALS' (Advanced Life Support) or 'BLS' (Basic)
    ambulance_eta_min: float
    rescue_team_id: str
    rescue_team_type: str
    hospital_id: str
    hospital_name: str
    hospital_trauma_level: str     # 'Level 1 Trauma', 'Level 2 Trauma', 'Community Care'
    hospital_distance_km: float
    allocation_score: float
    comparison_with_baseline: Dict[str, Any] = field(default_factory=dict)
    is_simulated: bool = True


@dataclass
class PatientRecord:
    patient_id: str
    age: int
    systolic_bp: int
    diastolic_bp: int
    spo2: int                      # % oxygen saturation
    heart_rate: int                # bpm
    respiratory_rate: int          # breaths per minute
    consciousness: str             # 'Alert', 'Voice', 'Pain', 'Unresponsive' (AVPU)
    symptoms: List[str] = field(default_factory=list)
    visible_trauma: str = "None"
    ambulatory: bool = False       # Able to walk (START criteria)


@dataclass
class RuleEvaluationResult:
    activated_rules: List[Dict[str, Any]] = field(default_factory=list)
    suggested_priority: str = "GREEN"  # 'RED', 'YELLOW', 'GREEN'
    escalation_triggered: bool = False
    rationale: str = ""


@dataclass
class BayesianEvaluationResult:
    hypotheses_posteriors: Dict[str, float] = field(default_factory=dict)
    top_condition: str = "Stable"
    top_probability: float = 0.0
    entropy: float = 0.0           # Measure of probabilistic uncertainty
    evidence_applied: List[str] = field(default_factory=list)


@dataclass
class TriageDecision:
    priority: str                  # 'RED', 'YELLOW', 'GREEN'
    priority_level: str            # 'Immediate (Priority 1)', 'Urgent (Priority 2)', 'Delayed (Priority 3)'
    rule_result: RuleEvaluationResult
    bayesian_result: BayesianEvaluationResult
    conservative_escalation_applied: bool
    explanation: str
    confidence_score: float
    disclaimer: str = MEDICAL_SAFETY_DISCLAIMER


@dataclass
class HospitalNotification:
    case_id: str
    timestamp: str
    triage_color: str
    patient_vitals_summary: Dict[str, Any]
    suspected_pathologies: List[Dict[str, Any]]
    assigned_transport: Dict[str, Any]
    hospital_id: str
    hospital_name: str
    sbar_formatted_text: str
    recommended_preparedness: List[str]
    disclaimer: str = MEDICAL_SAFETY_DISCLAIMER
    simulation_notice: str = SIMULATED_DATA_NOTICE


@dataclass
class IncidentState:
    """
    Central incident state shared across all modules in RescueMedAI.
    """
    incident_id: str = field(default_factory=lambda: f"INC-{uuid.uuid4().hex[:8].upper()}")
    timestamp: float = field(default_factory=time.time)
    disaster_type: str = "Earthquake"  # Earthquake, Flood, Urban Fire, Cyclone, Building Collapse
    location_name: str = "District Sector 4"
    incident_node: str = "N-04"
    
    # Stage 1 & 2: Vision & Severity
    image_name: Optional[str] = None
    image_path: Optional[str] = None
    vision_mode: str = "local_cv"      # 'local_cv' or 'demo_benchmark'
    detections: List[DetectionItem] = field(default_factory=list)
    road_conditions: List[RoadConditionAssessment] = field(default_factory=list)
    severity: Optional[SeverityAssessment] = None
    
    # Stage 3: Routing & Graph
    road_network_id: str = "metro_disaster_grid"
    blocked_edges: List[tuple] = field(default_factory=list)
    hazard_edges: Dict[tuple, float] = field(default_factory=dict)
    destination_node: str = "H-01"     # Target hospital node
    selected_route: Optional[RouteResult] = None
    route_benchmarks: Dict[str, RouteResult] = field(default_factory=dict)
    shortest_vs_safest: Optional[Dict[str, Any]] = None
    
    # Stage 4: Resource Allocation
    resource_assignment: Optional[ResourceAssignment] = None
    
    # Stage 5, 6, 7: Victim & Medical Decision Support
    patient: Optional[PatientRecord] = None
    triage: Optional[TriageDecision] = None
    
    # Stage 8: Hospital Notification
    hospital_notification: Optional[HospitalNotification] = None
    
    # Audit trail / execution log
    pipeline_log: List[Dict[str, Any]] = field(default_factory=list)

    def log_step(self, module_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Append an entry to the incident audit trail."""
        self.pipeline_log.append({
            "timestamp": time.time(),
            "module": module_name,
            "message": message,
            "details": details or {}
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to serializable dictionary for logs/JSON exports."""
        import dataclasses
        return dataclasses.asdict(self)
