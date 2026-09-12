"""
Unit tests for Routing module.
Verifies path validity, blockage avoidance, and A* vs Dijkstra cost optimality.
"""

import pytest
from src.backend.routing.graph import RoadNetwork
from src.backend.routing.search import a_star_search, dijkstra_search, bfs_search, greedy_best_first_search
from src.backend.routing.benchmark import RoutingBenchmarkEngine


def test_road_network_creation():
    net = RoadNetwork.create_default_city_grid()
    assert len(net.nodes) >= 15
    assert "D-01" in net.nodes
    assert "H-01" in net.nodes
    assert "N-04" in net.nodes


def test_a_star_and_dijkstra_optimal_cost():
    net = RoadNetwork.create_default_city_grid()
    start = "N-04"
    goal = "H-01"

    a_star_res = a_star_search(net, start, goal)
    dijkstra_res = dijkstra_search(net, start, goal)

    assert a_star_res.path, "A* must find a path"
    assert dijkstra_res.path, "Dijkstra must find a path"
    assert a_star_res.path[0] == start
    assert a_star_res.path[-1] == goal
    assert dijkstra_res.path[0] == start
    assert dijkstra_res.path[-1] == goal

    # Check that both find the exact same optimal cost under the admissible heuristic
    assert pytest.approx(a_star_res.cost, rel=1e-4) == dijkstra_res.cost


def test_blocked_road_avoidance():
    net = RoadNetwork.create_default_city_grid()
    start = "N-04"
    goal = "H-01"

    # Get baseline path
    base_res = a_star_search(net, start, goal)
    assert len(base_res.path) >= 2

    # Block the primary direct connection out of N-04
    first_step = base_res.path[1]
    net.set_edge_status("N-04", first_step, is_blocked=True)

    # Rerun A*
    reroute_res = a_star_search(net, start, goal)
    assert reroute_res.path, "A* must find an alternate path"
    assert reroute_res.path[1] != first_step, "A* must not use the blocked road"
    assert RoutingBenchmarkEngine.is_path_valid(net, reroute_res.path)


def test_benchmark_engine():
    net = RoadNetwork.create_default_city_grid()
    # Add a blocked road and a hazard
    net.set_edge_status("N-05", "N-06", is_blocked=True)
    net.set_edge_status("N-02", "N-03", is_blocked=False, hazard_weight=1.5)

    bench = RoutingBenchmarkEngine.run_benchmark(net, "N-04", "H-01")

    assert bench["verification"]["both_found"]
    assert bench["verification"]["optimal_cost_matched"]
    assert bench["verification"]["all_valid_unblocked"]
    assert len(bench["summary_table"]) == 4

    # Verify that all 4 algorithms record factual non-negative metrics
    for row in bench["summary_table"]:
        assert row["Nodes Expanded"] > 0
        assert row["Runtime (ms)"] >= 0.0


def test_safest_route_differs_from_shortest_under_hazard():
    from src.backend.routing.search import compare_shortest_vs_safest
    net = RoadNetwork.create_default_city_grid()
    start = "N-08"
    goal = "H-02"

    # Heavily penalize the direct corridor with flood hazard
    net.set_edge_status("N-08", "N-09", is_blocked=False, hazard_weight=5.0)

    cmp = compare_shortest_vs_safest(net, start, goal)
    assert cmp["is_safest_different_from_shortest"] is True
    # The safest A* path costs less risk time than the naive shortest path
    assert cmp["safest_route"].cost < cmp["shortest_route"].cost
    # The safest route traverses a longer physical distance to avoid the hazard
    assert cmp["safest_route"].distance_km >= cmp["shortest_route"].distance_km
    assert cmp["risk_cost_saved_min"] > 0

