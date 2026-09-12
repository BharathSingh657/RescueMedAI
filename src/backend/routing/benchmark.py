"""
Routing Benchmark Engine.
Executes A*, Dijkstra (UCS), BFS, and Greedy Best-First Search on identical network state,
recording factual empirical observations without fabricated values.
"""

from typing import Dict, Any, List
import math
from src.backend.routing.graph import RoadNetwork
from src.backend.routing.search import a_star_search, dijkstra_search, bfs_search, greedy_best_first_search
from src.backend.state import RouteResult


class RoutingBenchmarkEngine:
    @staticmethod
    def is_path_valid(graph: RoadNetwork, path: List[str]) -> bool:
        """Verifies that each step is a valid, unblocked edge."""
        if not path or len(path) < 2:
            return False
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u not in graph.adj or v not in graph.adj[u]:
                return False
            if graph.adj[u][v]["is_blocked"]:
                return False
        return True

    @staticmethod
    def run_benchmark(graph: RoadNetwork, start: str, goal: str) -> Dict[str, Any]:
        """
        Executes all 4 search algorithms on the same graph instance.
        Returns empirical results and factual verifications.
        """
        results: Dict[str, RouteResult] = {
            "A*": a_star_search(graph, start, goal),
            "Dijkstra": dijkstra_search(graph, start, goal),
            "BFS": bfs_search(graph, start, goal),
            "Greedy Best-First": greedy_best_first_search(graph, start, goal)
        }

        # Verification metrics
        a_star_res = results["A*"]
        dijkstra_res = results["Dijkstra"]

        both_found = bool(a_star_res.path and dijkstra_res.path)
        cost_difference = abs(a_star_res.cost - dijkstra_res.cost) if both_found else float('nan')
        optimal_cost_matched = bool(both_found and math.isclose(a_star_res.cost, dijkstra_res.cost, rel_tol=1e-5))

        # Check blocked roads avoidance for all found paths
        validity = {
            algo: RoutingBenchmarkEngine.is_path_valid(graph, res.path) if res.path else False
            for algo, res in results.items()
        }

        summary_table = []
        for algo, res in results.items():
            summary_table.append({
                "Algorithm": algo,
                "Path Found": bool(res.path),
                "Path (hops)": len(res.path) - 1 if res.path else 0,
                "Cost": round(res.cost, 2) if not math.isinf(res.cost) else "Unreachable",
                "Distance (km)": round(res.distance_km, 2),
                "Travel Time (min)": round(res.travel_time_min, 2),
                "Nodes Expanded": res.nodes_expanded,
                "Runtime (ms)": round(res.runtime_ms, 3),
                "Valid & Unblocked": validity.get(algo, False)
            })

        return {
            "results": results,
            "summary_table": summary_table,
            "verification": {
                "both_found": both_found,
                "optimal_cost_matched": optimal_cost_matched,
                "cost_difference": cost_difference,
                "all_valid_unblocked": all(validity.values())
            }
        }
