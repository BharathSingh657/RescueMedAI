"""
Disaster Severity Estimation Engine.
Calculates a transparent multi-factor score from visual damage and road obstructions.
"""

from typing import List, Dict, Any, Optional
from src.backend.state import DetectionItem, SeverityAssessment, RoadConditionAssessment


class DisasterSeverityEstimator:
    """
    Combines computer vision observations and segmented road conditions into an explainable disaster severity assessment.
    """

    @staticmethod
    def estimate(
        detections: List[DetectionItem],
        road_conditions: Optional[List[RoadConditionAssessment]] = None
    ) -> SeverityAssessment:
        damaged_building_count = sum(1 for d in detections if d.label == "damaged_building")
        blocked_road_count = sum(1 for d in detections if d.label in ["blocked_road", "flood_inundation"])
        fire_detected = any(d.label == "fire_indicator" for d in detections)

        # If detailed road conditions are provided, supplement blocked count
        road_summary_parts = []
        if road_conditions:
            impassable_count = sum(1 for rc in road_conditions if rc.is_impassable or rc.condition in ["Blocked", "Flooded", "Severely Damaged"])
            blocked_road_count = max(blocked_road_count, impassable_count)
            for rc in road_conditions:
                if rc.condition != "Normal":
                    road_summary_parts.append(f"{rc.segment_id}: {rc.condition} ({rc.confidence:.0%})")

        # Normalize metrics against operational disaster thresholds
        damaged_building_ratio = min(1.0, damaged_building_count / 4.0)
        blocked_road_norm = min(1.0, blocked_road_count / 3.0)
        fire_factor = 1.0 if fire_detected else 0.0
        zone_density = round(min(1.0, (damaged_building_count + blocked_road_count) / 6.0), 2)

        # Transparent weighted scoring formula:
        # Score = 0.35 * damage_ratio + 0.25 * blocked_roads + 0.20 * fire + 0.20 * zone_density
        score = (
            0.35 * damaged_building_ratio +
            0.25 * blocked_road_norm +
            0.20 * fire_factor +
            0.20 * zone_density
        )
        score = round(min(1.0, max(0.0, score)), 2)

        # Classification
        if score >= 0.75:
            label = "Critical"
            rationale = "Widespread structural collapse, active fire, or extensive arterial road blockages."
        elif score >= 0.50:
            label = "High"
            rationale = "Significant structural impairment and multiple road obstructions requiring rerouting."
        elif score >= 0.25:
            label = "Medium"
            rationale = "Localized debris or single obstruction; secondary arterial routes remain navigable."
        else:
            label = "Low"
            rationale = "Minimal damage signatures; infrastructure intact with negligible operational risk."

        road_summary_text = " | ".join(road_summary_parts) if road_summary_parts else "All surveyed corridors passable"

        return SeverityAssessment(
            score=score,
            label=label,
            damaged_building_ratio=round(damaged_building_ratio, 2),
            blocked_road_count=blocked_road_count,
            fire_detected=fire_detected,
            affected_zone_density=zone_density,
            rationale=rationale,
            road_condition_summary=road_summary_text
        )
