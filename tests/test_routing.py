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


def test_katpadi_vellore_map_routing():
    presets = RoadNetwork.get_available_presets()
    assert "katpadi_vellore" in presets

    net = RoadNetwork.create_from_preset("katpadi_vellore")
    assert "K-01" in net.nodes
    assert "H-01" in net.nodes

    # Route from Depot D-01 to Government Hospital H-01
    route = a_star_search(net, "D-01", "H-01")
    assert route.path
    assert route.path[0] == "D-01"
    assert route.path[-1] == "H-01"
    assert route.cost > 0.0

    # Block direct CM John St -> Vallimalai Rd corridor (K-05 -> K-06)
    net.set_edge_status("K-05", "K-06", is_blocked=True)
    reroute = a_star_search(net, "D-01", "H-01")
    assert reroute.path
    assert ("K-05", "K-06") not in zip(reroute.path, reroute.path[1:])
    assert ("K-06", "K-05") not in zip(reroute.path, reroute.path[1:])


def test_chatgpt_json_map_parsing():
    chatgpt_json = """
    {
      "map_name": "RescueMedAI Prototype Map",
      "nodes": [
        {"id": "N1", "x": 100, "y": 100},
        {"id": "N2", "x": 250, "y": 100},
        {"id": "N3", "x": 400, "y": 100},
        {"id": "N4", "x": 100, "y": 250},
        {"id": "N5", "x": 250, "y": 250},
        {"id": "N6", "x": 400, "y": 250}
      ],
      "roads": [
        {"id": "R1", "from": "N1", "to": "N2", "distance": 150},
        {"id": "R2", "from": "N2", "to": "N3", "distance": 150},
        {"id": "R3", "from": "N1", "to": "N4", "distance": 150},
        {"id": "R4", "from": "N2", "to": "N5", "distance": 150},
        {"id": "R5", "from": "N3", "to": "N6", "distance": 150}
      ]
    }
    """
    net = RoadNetwork.from_json_string(chatgpt_json)
    assert len(net.nodes) == 6
    assert "N1" in net.nodes
    assert "N6" in net.nodes
    
    # Verify coordinates normalized to 0..10 grid
    assert net.nodes["N1"]["x"] == 0.0
    assert net.nodes["N6"]["x"] == 10.0

    # Verify A* search works on uploaded ChatGPT map
    route = a_star_search(net, "N1", "N6")
    assert route.path
    assert route.path[0] == "N1"
    assert route.path[-1] == "N6"



