"""
RescueMedAI Backend Core Package.
Provides algorithms and models for Road Networks, Hazard Updates, A* Routing, Vision Perception,
Triage, Resource Allocation, and Hospital Notification.
"""

from src.backend.state import (
    IncidentState,
    PatientRecord,
    RouteResult,
    DetectionItem,
    SeverityAssessment,
    RoadConditionAssessment,
    ResourceAssignment,
    TriageDecision,
    HospitalNotification,
    MEDICAL_SAFETY_DISCLAIMER,
    SIMULATED_DATA_NOTICE
)
from src.backend.routing.graph import RoadNetwork
from src.backend.routing.search import a_star_search
from src.backend.pipeline import RescueMedPipeline

__all__ = [
    "IncidentState",
    "PatientRecord",
    "RouteResult",
    "DetectionItem",
    "SeverityAssessment",
    "RoadConditionAssessment",
    "ResourceAssignment",
    "TriageDecision",
    "HospitalNotification",
    "MEDICAL_SAFETY_DISCLAIMER",
    "SIMULATED_DATA_NOTICE",
    "RoadNetwork",
    "a_star_search",
    "RescueMedPipeline",
]
