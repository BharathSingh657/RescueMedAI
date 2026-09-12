"""
Search algorithms for RescueMedAI:
- A* Search (admissible heuristic + dynamic hazard cost)
- Dijkstra / Uniform Cost Search
- Breadth-First Search (BFS)
- Greedy Best-First Search
"""

import heapq
from collections import deque
import time
from typing import List, Tuple, Dict, Optional, Any
from src.backend.routing.graph import RoadNetwork
from src.backend.state import RouteResult


def _reconstruct_path(came_from: Dict[str, str], current: str) -> List[str]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _calculate_path_metrics(graph: RoadNetwork, path: List[str]) -> Tuple[float, float, float]:
    """
    Computes (cost, total_distance_km, total_base_time_min) for a valid path.
    """
    if len(path) < 2:
        return 0.0, 0.0, 0.0

    cost = 0.0
    dist_km = 0.0
    time_min = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = graph.adj[u][v]
        edge_cost = graph.get_edge_cost(u, v)
        cost += edge_cost
        dist_km += edge["distance_km"]
        time_min += edge["base_time_min"]

    return cost, dist_km, time_min


def a_star_search(graph: RoadNetwork, start: str, goal: str) -> RouteResult:
    """
    A* Search evaluating f(n) = g(n) + h(n) with admissible travel heuristic.
    """
    start_time = time.perf_counter()
    if start not in graph.nodes or goal not in graph.nodes:
        return RouteResult("A*", [], float('inf'), 0.0, 0.0, 0, 0.0, False)

    frontier = []
    tie_breaker = 0
    h_start = graph.admissible_travel_heuristic(start, goal)
    heapq.heappush(frontier, (h_start, tie_breaker, start))

    came_from: Dict[str, str] = {}
    g_score: Dict[str, float] = {start: 0.0}
    visited = set()
    nodes_expanded = 0

    while frontier:
        f, _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            path = _reconstruct_path(came_from, current)
            cost, dist_km, time_min = _calculate_path_metrics(graph, path)
            runtime_ms = (time.perf_counter() - start_time) * 1000.0
            return RouteResult(
                algorithm="A*",
                path=path,
                cost=cost,
                distance_km=dist_km,
                travel_time_min=time_min,
                nodes_expanded=nodes_expanded,
                runtime_ms=runtime_ms,
                is_optimal=True
            )

        for neighbor in graph.get_neighbors(current):
            edge_cost = graph.get_edge_cost(current, neighbor)
            tentative_g = g_score[current] + edge_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = graph.admissible_travel_heuristic(neighbor, goal)
                tie_breaker += 1
                heapq.heappush(frontier, (tentative_g + h, tie_breaker, neighbor))

    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    return RouteResult("A*", [], float('inf'), 0.0, 0.0, nodes_expanded, runtime_ms, False)


def dijkstra_search(graph: RoadNetwork, start: str, goal: str) -> RouteResult:
    """
    Dijkstra / Uniform Cost Search evaluating f(n) = g(n) without heuristic.
    """
    start_time = time.perf_counter()
    if start not in graph.nodes or goal not in graph.nodes:
        return RouteResult("Dijkstra (UCS)", [], float('inf'), 0.0, 0.0, 0, 0.0, False)

    frontier = []
    tie_breaker = 0
    heapq.heappush(frontier, (0.0, tie_breaker, start))

    came_from: Dict[str, str] = {}
    g_score: Dict[str, float] = {start: 0.0}
    visited = set()
    nodes_expanded = 0

    while frontier:
        cost_so_far, _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            path = _reconstruct_path(came_from, current)
            cost, dist_km, time_min = _calculate_path_metrics(graph, path)
            runtime_ms = (time.perf_counter() - start_time) * 1000.0
            return RouteResult(
                algorithm="Dijkstra (UCS)",
                path=path,
                cost=cost,
                distance_km=dist_km,
                travel_time_min=time_min,
                nodes_expanded=nodes_expanded,
                runtime_ms=runtime_ms,
                is_optimal=True
            )

        for neighbor in graph.get_neighbors(current):
            edge_cost = graph.get_edge_cost(current, neighbor)
            tentative_g = g_score[current] + edge_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                tie_breaker += 1
                heapq.heappush(frontier, (tentative_g, tie_breaker, neighbor))

    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    return RouteResult("Dijkstra (UCS)", [], float('inf'), 0.0, 0.0, nodes_expanded, runtime_ms, False)


def bfs_search(graph: RoadNetwork, start: str, goal: str) -> RouteResult:
    """
    Breadth-First Search (BFS) finding path with fewest edge hops.
    """
    start_time = time.perf_counter()
    if start not in graph.nodes or goal not in graph.nodes:
        return RouteResult("BFS", [], float('inf'), 0.0, 0.0, 0, 0.0, False)

    queue = deque([start])
    visited = {start}
    came_from: Dict[str, str] = {}
    nodes_expanded = 0

    while queue:
        current = queue.popleft()
        nodes_expanded += 1

        if current == goal:
            path = _reconstruct_path(came_from, current)
            cost, dist_km, time_min = _calculate_path_metrics(graph, path)
            runtime_ms = (time.perf_counter() - start_time) * 1000.0
            return RouteResult(
                algorithm="BFS",
                path=path,
                cost=cost,
                distance_km=dist_km,
                travel_time_min=time_min,
                nodes_expanded=nodes_expanded,
                runtime_ms=runtime_ms,
                is_optimal=False  # Fewest hops is not necessarily lowest cost
            )

        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    return RouteResult("BFS", [], float('inf'), 0.0, 0.0, nodes_expanded, runtime_ms, False)


def greedy_best_first_search(graph: RoadNetwork, start: str, goal: str) -> RouteResult:
    """
    Greedy Best-First Search evaluating f(n) = h(n) (heuristic distance only).
    """
    start_time = time.perf_counter()
    if start not in graph.nodes or goal not in graph.nodes:
        return RouteResult("Greedy Best-First", [], float('inf'), 0.0, 0.0, 0, 0.0, False)

    frontier = []
    tie_breaker = 0
    h_start = graph.euclidean_distance(start, goal)
    heapq.heappush(frontier, (h_start, tie_breaker, start))

    came_from: Dict[str, str] = {}
    visited = set()
    nodes_expanded = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            path = _reconstruct_path(came_from, current)
            cost, dist_km, time_min = _calculate_path_metrics(graph, path)
            runtime_ms = (time.perf_counter() - start_time) * 1000.0
            return RouteResult(
                algorithm="Greedy Best-First",
                path=path,
                cost=cost,
                distance_km=dist_km,
                travel_time_min=time_min,
                nodes_expanded=nodes_expanded,
                runtime_ms=runtime_ms,
                is_optimal=False
            )

        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                came_from[neighbor] = current
                h = graph.euclidean_distance(neighbor, goal)
                tie_breaker += 1
                heapq.heappush(frontier, (h, tie_breaker, neighbor))

    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    return RouteResult("Greedy Best-First", [], float('inf'), 0.0, 0.0, nodes_expanded, runtime_ms, False)


def shortest_distance_search(graph: RoadNetwork, start: str, goal: str) -> RouteResult:
    """
    Computes the shortest physical distance route (ignoring hazard penalties).
    Used as an operational baseline to prove that the safest route is not always the shortest.
    """
    start_time = time.perf_counter()
    if start not in graph.nodes or goal not in graph.nodes:
        return RouteResult("Shortest-Distance Baseline", [], float('inf'), 0.0, 0.0, 0, 0.0, False)

    frontier = []
    tie_breaker = 0
    heapq.heappush(frontier, (0.0, tie_breaker, start))

    came_from: Dict[str, str] = {}
    dist_so_far: Dict[str, float] = {start: 0.0}
    visited = set()
    nodes_expanded = 0

    while frontier:
        d, _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            path = _reconstruct_path(came_from, current)
            cost, dist_km, time_min = _calculate_path_metrics(graph, path)
            runtime_ms = (time.perf_counter() - start_time) * 1000.0
            return RouteResult(
                algorithm="Shortest-Distance Baseline",
                path=path,
                cost=cost,
                distance_km=dist_km,
                travel_time_min=time_min,
                nodes_expanded=nodes_expanded,
                runtime_ms=runtime_ms,
                is_optimal=False
            )

        for neighbor in graph.get_neighbors(current):
            edge_dist = graph.adj[current][neighbor]["distance_km"]
            tentative_dist = dist_so_far[current] + edge_dist

            if tentative_dist < dist_so_far.get(neighbor, float('inf')):
                came_from[neighbor] = current
                dist_so_far[neighbor] = tentative_dist
                tie_breaker += 1
                heapq.heappush(frontier, (tentative_dist, tie_breaker, neighbor))

    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    return RouteResult("Shortest-Distance Baseline", [], float('inf'), 0.0, 0.0, nodes_expanded, runtime_ms, False)


def compare_shortest_vs_safest(graph: RoadNetwork, start: str, goal: str) -> Dict[str, Any]:
    """
    Directly compares the naive Shortest-Distance route against the A* Hazard-Weighted Safest route.
    Demonstrates that minimizing physical distance alone in a disaster leads to dangerous, high-risk routes.
    """
    shortest_res = shortest_distance_search(graph, start, goal)
    safest_res = a_star_search(graph, start, goal)

    is_different = (shortest_res.path != safest_res.path)
    dist_diff = round(safest_res.distance_km - shortest_res.distance_km, 2)
    risk_time_saved = round(shortest_res.cost - safest_res.cost, 2)

    if is_different and risk_time_saved > 0:
        explanation = (
            f"The naive shortest route ({' ➔ '.join(shortest_res.path)}) covers {shortest_res.distance_km:.1f} km "
            f"but traverses severe hazard/flood zones, resulting in an effective risk cost of {shortest_res.cost:.1f} min. "
            f"A* identifies a safer detour ({' ➔ '.join(safest_res.path)}) that is {dist_diff:+.1f} km longer, "
            f"yet cuts disaster risk traversal time by {risk_time_saved:.1f} minutes ({risk_time_saved / max(0.1, shortest_res.cost):.0%} reduction)."
        )
    elif is_different:
        explanation = (
            f"A* selected an alternate path ({' ➔ '.join(safest_res.path)}) differing from the shortest distance "
            f"path ({' ➔ '.join(shortest_res.path)}) to balance road speed limits and terrain feasibility."
        )
    else:
        explanation = (
            f"The safest route coincides with the shortest route ({' ➔ '.join(safest_res.path)}) "
            f"because no severe hazards or road blockages compromised the direct corridor."
        )

    return {
        "shortest_route": shortest_res,
        "safest_route": safest_res,
        "is_safest_different_from_shortest": is_different,
        "distance_difference_km": dist_diff,
        "risk_cost_saved_min": risk_time_saved,
        "explanation": explanation
    }
