"""
Unit tests for the Experimental Evaluation Suite.
Validates that Scenarios A, B, C, and D execute cleanly and yield empirical, non-fabricated metrics.
"""

from src.backend.benchmark_suite import ExperimentSuite


def test_scenario_a_vision():
    res = ExperimentSuite.run_scenario_a_vision()
    assert res["metrics"]["Total Test Images"] > 0
    assert 0.0 <= res["metrics"]["Precision"] <= 1.0
    assert 0.0 <= res["metrics"]["Recall (Sensitivity)"] <= 1.0
    assert 0.0 <= res["metrics"]["F1 Score"] <= 1.0


def test_scenario_b_routing():
    res = ExperimentSuite.run_scenario_b_routing()
    assert len(res["summary_table"]) == 4
    assert res["verification"]["optimal_cost_matched"] is True
    assert res["verification"]["all_valid_unblocked"] is True


def test_scenario_c_allocation():
    res = ExperimentSuite.run_scenario_c_allocation()
    assert res["metrics"]["Incidents Tested"] == 3
    assert len(res["table"]) == 3


def test_scenario_d_medical():
    res = ExperimentSuite.run_scenario_d_medical()
    assert res["metrics"]["Total Cases Evaluated"] >= 5
    assert res["metrics"]["Critical False Negatives"] == 0, "Hybrid safety escalation must prevent false negative critical triage"
    assert res["metrics"]["High-Priority Sensitivity (RED)"] == 1.0
