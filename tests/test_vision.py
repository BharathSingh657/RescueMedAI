"""
Unit tests for Computer Vision and Severity modules.
Verifies real local pixel analysis, benchmark annotation ingestion, and transparent severity scoring.
"""

import os
from PIL import Image
from src.backend.vision.detector import DisasterVisionDetector
from src.backend.vision.severity import DisasterSeverityEstimator


def test_local_pixel_analysis_fire():
    img_path = "data/sample_images/urban_fire.jpg"
    assert os.path.exists(img_path)
    img = Image.open(img_path)

    detections = DisasterVisionDetector.detect_hazards(img)
    assert any(d.label in ["fire_indicator", "damaged_building", "blocked_road"] for d in detections)
    assert all("PRETRAINED DEEP LEARNING" in d.source or "BENCHMARK" in d.source for d in detections)


def test_road_condition_assessment_flood():
    img_path = "data/sample_images/flash_flood.jpg"
    assert os.path.exists(img_path)
    img = Image.open(img_path)

    road_conditions = DisasterVisionDetector.assess_road_conditions(img)
    assert len(road_conditions) == 3
    # At least one corridor must be classified as Flooded
    assert any(rc.condition == "Flooded" for rc in road_conditions)
    flooded_rc = next(rc for rc in road_conditions if rc.condition == "Flooded")
    assert flooded_rc.confidence > 0.75
    assert flooded_rc.flood_coverage_pct > 15.0
    assert flooded_rc.is_impassable is True


def test_benchmark_annotations_ingestion():
    img_path = "data/sample_images/earthquake_collapse.jpg"
    img = Image.open(img_path)

    mock_benchmarks = [
        {"label": "damaged_building", "confidence": 0.95, "bbox": [50, 50, 200, 200]},
        {"label": "blocked_road", "confidence": 0.91, "bbox": [100, 300, 500, 450]}
    ]
    detections = DisasterVisionDetector.detect_hazards(img, benchmark_annotations=mock_benchmarks)
    assert len(detections) == 2
    assert all("BENCHMARK" in d.source for d in detections)


def test_severity_estimation_critical():
    img_path = "data/sample_images/earthquake_collapse.jpg"
    img = Image.open(img_path)

    mock_benchmarks = [
        {"label": "damaged_building", "confidence": 0.95, "bbox": [50, 50, 200, 200]},
        {"label": "damaged_building", "confidence": 0.92, "bbox": [250, 50, 450, 200]},
        {"label": "blocked_road", "confidence": 0.91, "bbox": [100, 300, 500, 450]},
        {"label": "blocked_road", "confidence": 0.88, "bbox": [20, 250, 200, 350]},
        {"label": "fire_indicator", "confidence": 0.85, "bbox": [300, 100, 350, 150]}
    ]
    detections = DisasterVisionDetector.detect_hazards(img, benchmark_annotations=mock_benchmarks)
    assessment = DisasterSeverityEstimator.estimate(detections)

    assert assessment.score >= 0.55
    assert assessment.label == "High"
    assert assessment.fire_detected is True
    assert assessment.blocked_road_count == 2

    # Add more catastrophic damage to test Critical (> 0.75)
    critical_benchmarks = mock_benchmarks + [
        {"label": "damaged_building", "confidence": 0.96, "bbox": [50, 50, 100, 100]},
        {"label": "damaged_building", "confidence": 0.94, "bbox": [100, 50, 150, 100]},
        {"label": "blocked_road", "confidence": 0.95, "bbox": [10, 10, 50, 50]},
        {"label": "blocked_road", "confidence": 0.93, "bbox": [50, 10, 100, 50]}
    ]
    crit_detections = DisasterVisionDetector.detect_hazards(img, benchmark_annotations=critical_benchmarks)
    crit_assessment = DisasterSeverityEstimator.estimate(crit_detections)
    assert crit_assessment.score >= 0.75
    assert crit_assessment.label == "Critical"
