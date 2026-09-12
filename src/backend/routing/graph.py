"""
Road Graph representation for disaster response.
Supports dynamic hazard-weighting, road closures, and spatial coordinates.
"""

from typing import Dict, List, Tuple, Optional, Any
import math
import json


class RoadNetwork:
    def __init__(self):
        # nodes: id -> {"name": str, "x": float, "y": float, "type": str}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # adjacency: u -> {v: {"distance_km": float, "base_speed_kmh": float, "hazard_weight": float, "is_blocked": bool}}
        self.adj: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def add_node(self, node_id: str, name: str, x: float, y: float, node_type: str = "intersection"):
        self.nodes[node_id] = {
            "name": name,
            "x": float(x),
            "y": float(y),
            "type": node_type
        }
        if node_id not in self.adj:
            self.adj[node_id] = {}

    def add_edge(
        self,
        u: str,
        v: str,
        distance_km: float,
        base_speed_kmh: float = 40.0,
        hazard_weight: float = 0.0,
        is_blocked: bool = False,
        bidirectional: bool = True
    ):
        base_time_min = (distance_km / base_speed_kmh) * 60.0
        edge_data = {
            "distance_km": float(distance_km),
            "base_speed_kmh": float(base_speed_kmh),
            "base_time_min": float(base_time_min),
            "hazard_weight": max(0.0, float(hazard_weight)),
            "is_blocked": bool(is_blocked)
        }
        if u not in self.adj:
            self.adj[u] = {}
        self.adj[u][v] = dict(edge_data)

        if bidirectional:
            if v not in self.adj:
                self.adj[v] = {}
            self.adj[v][u] = dict(edge_data)

    def set_edge_status(self, u: str, v: str, is_blocked: bool, hazard_weight: Optional[float] = None):
        """Update road blockage or hazard status."""
        for a, b in [(u, v), (v, u)]:
            if a in self.adj and b in self.adj[a]:
                self.adj[a][b]["is_blocked"] = is_blocked
                if hazard_weight is not None:
                    self.adj[a][b]["hazard_weight"] = max(0.0, float(hazard_weight))

    def apply_incident_hazard(self, center_node: str, radius_km: float = 3.0, hazard_factor: float = 1.5):
        """
        Dynamically penalize edges located within proximity to a severe incident center.
        """
        if center_node not in self.nodes:
            return
        cx, cy = self.nodes[center_node]["x"], self.nodes[center_node]["y"]
        
        for u in self.adj:
            for v, edge in self.adj[u].items():
                ux, uy = self.nodes[u]["x"], self.nodes[u]["y"]
                vx, vy = self.nodes[v]["x"], self.nodes[v]["y"]
                mid_x, mid_y = (ux + vx) / 2.0, (uy + vy) / 2.0
                dist_to_center = math.hypot(mid_x - cx, mid_y - cy)
                
                if dist_to_center <= radius_km:
                    penalty = hazard_factor * (1.0 - (dist_to_center / radius_km))
                    edge["hazard_weight"] = max(edge["hazard_weight"], penalty)

    def get_edge_cost(self, u: str, v: str) -> float:
        """
        Computes the effective traversal cost:
        cost = base_time_min * (1.0 + hazard_weight)
        Returns infinity if road is impassable / blocked.
        """
        if u not in self.adj or v not in self.adj[u]:
            return float('inf')
        edge = self.adj[u][v]
        if edge["is_blocked"]:
            return float('inf')
        return edge["base_time_min"] * (1.0 + edge["hazard_weight"])

    def get_neighbors(self, u: str) -> List[str]:
        """Returns reachable neighbors that are not blocked."""
        if u not in self.adj:
            return []
        return [v for v, data in self.adj[u].items() if not data["is_blocked"]]

    def euclidean_distance(self, u: str, v: str) -> float:
        """Euclidean distance in coordinate space (km)."""
        if u not in self.nodes or v not in self.nodes:
            return 0.0
        n1 = self.nodes[u]
        n2 = self.nodes[v]
        return math.hypot(n1["x"] - n2["x"], n1["y"] - n2["y"])

    def admissible_travel_heuristic(self, u: str, goal: str, max_speed_kmh: float = 60.0) -> float:
        """
        Admissible lower bound for travel time in minutes:
        h(u) = (Euclidean distance to goal / max possible speed) * 60
        Guaranteed h(u) <= true optimal cost because:
        1. Straight line distance <= actual graph distance.
        2. Max speed >= any actual road speed.
        3. Hazard weight is non-negative (hazard >= 0).
        Thus h(u) is provably admissible and consistent.
        """
        dist_km = self.euclidean_distance(u, goal)
        return (dist_km / max_speed_kmh) * 60.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [
                {
                    "u": u,
                    "v": v,
                    **data
                }
                for u in self.adj
                for v, data in self.adj[u].items()
                if u < v  # deduplicate undirected edges
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoadNetwork":
        net = cls()
        for node_id, node_info in data["nodes"].items():
            net.add_node(node_id, node_info["name"], node_info["x"], node_info["y"], node_info.get("type", "intersection"))
        for edge in data["edges"]:
            net.add_edge(
                edge["u"],
                edge["v"],
                edge["distance_km"],
                edge.get("base_speed_kmh", 40.0),
                edge.get("hazard_weight", 0.0),
                edge.get("is_blocked", False),
                bidirectional=True
            )
        return net

    @classmethod
    def create_default_city_grid(cls) -> "RoadNetwork":
        """
        Creates a realistic 16-node disaster response road network.
        Nodes include:
        - Disaster sectors (N-01 to N-12)
        - Ambulance Depot (D-01)
        - Hospital 1: Memorial Trauma Center (H-01)
        - Hospital 2: Metro General (H-02)
        - Hospital 3: St. Jude Community Hospital (H-03)
        """
        net = cls()
        # 4x4 Grid coordinates (km)
        coords = {
            "D-01": ("Central EMS Depot", 0.0, 0.0, "depot"),
            "N-01": ("Northwest Sector", 0.0, 2.5, "intersection"),
            "N-02": ("North Bridge Inbound", 2.5, 2.5, "intersection"),
            "N-03": ("River Crossing Bridge", 5.0, 2.5, "bridge"),
            "H-01": ("Memorial Level-1 Trauma Center", 7.5, 2.5, "hospital"),
            
            "N-04": ("West Market Square (Incident)", 0.0, 5.0, "incident_zone"),
            "N-05": ("City Center Junction", 2.5, 5.0, "intersection"),
            "N-06": ("East Commercial Arterial", 5.0, 5.0, "intersection"),
            "N-07": ("East Ring Highway", 7.5, 5.0, "intersection"),
            
            "N-08": ("Southwest Industrial Area", 0.0, 7.5, "intersection"),
            "N-09": ("Grand Central Avenue", 2.5, 7.5, "intersection"),
            "H-02": ("Metro General Hospital", 5.0, 7.5, "hospital"),
            "N-10": ("Southeast Transit Corridor", 7.5, 7.5, "intersection"),
            
            "N-11": ("Outer South Perimeter", 0.0, 10.0, "intersection"),
            "N-12": ("South Boulevard", 2.5, 10.0, "intersection"),
            "H-03": ("St. Jude Community Hospital", 5.0, 10.0, "hospital"),
        }

        for nid, (name, x, y, ntype) in coords.items():
            net.add_node(nid, name, x, y, ntype)

        # Standard grid edges (distance_km, speed_kmh)
        edges = [
            # Row 0
            ("D-01", "N-01", 2.5, 45.0),
            ("N-01", "N-02", 2.5, 40.0),
            ("N-02", "N-03", 2.5, 35.0),
            ("N-03", "H-01", 2.5, 50.0),
            
            # Row 1
            ("N-04", "N-05", 2.5, 40.0),
            ("N-05", "N-06", 2.5, 45.0),
            ("N-06", "N-07", 2.5, 50.0),
            
            # Row 2
            ("N-08", "N-09", 2.5, 40.0),
            ("N-09", "H-02", 2.5, 40.0),
            ("H-02", "N-10", 2.5, 45.0),
            
            # Row 3
            ("N-11", "N-12", 2.5, 35.0),
            ("N-12", "H-03", 2.5, 35.0),

            # Vertical connections
            ("D-01", "N-04", 5.0, 45.0),
            ("N-04", "N-08", 2.5, 35.0),
            ("N-08", "N-11", 2.5, 35.0),
            
            ("N-01", "N-05", 2.5, 40.0),
            ("N-05", "N-09", 2.5, 45.0),
            ("N-09", "N-12", 2.5, 35.0),
            
            ("N-02", "N-06", 2.5, 40.0),
            ("N-06", "H-02", 2.5, 40.0),
            ("H-02", "H-03", 2.5, 35.0),
            
            ("N-03", "N-07", 2.5, 45.0),
            ("N-07", "N-10", 2.5, 50.0),
            
            # Diagonal highway bypass
            ("D-01", "N-05", 3.5, 60.0),
            ("N-05", "H-01", 5.6, 60.0),
        ]

        for u, v, dist, speed in edges:
            net.add_edge(u, v, dist, speed)

        return net
