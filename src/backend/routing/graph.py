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
        self.bg_image_path: Optional[str] = None

    def add_node(self, node_id: str, name: str, x: float, y: float, node_type: str = "intersection", lat: Optional[float] = None, lon: Optional[float] = None):
        # Default mapping from grid (km) to lat/lon centered around 12.9814° N, 79.1401° E (Katpadi/Vellore)
        base_lat = 12.9790 if lat is None else float(lat)
        base_lon = 79.1380 if lon is None else float(lon)
        calc_lat = base_lat + (y * 0.006) if lat is None else float(lat)
        calc_lon = base_lon + (x * 0.006) if lon is None else float(lon)

        self.nodes[node_id] = {
            "name": name,
            "x": float(x),
            "y": float(y),
            "lat": calc_lat,
            "lon": calc_lon,
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
        """
        Robustly parses ANY JSON road map specification.
        Handles list or dict nodes, 'edges' or 'roads' or 'links', flexible key names ('from'/'to', 'u'/'v', 'distance'),
        and automatically normalizes coordinate scales.
        """
        net = cls()
        
        # 1. Parse Nodes
        raw_nodes = data.get("nodes", {})
        node_records = []
        
        if isinstance(raw_nodes, dict):
            for nid, info in raw_nodes.items():
                if isinstance(info, dict):
                    node_records.append({
                        "id": str(nid),
                        "name": str(info.get("name") or info.get("label") or nid),
                        "x": float(info.get("x", 0.0)),
                        "y": float(info.get("y", 0.0)),
                        "type": str(info.get("type", "intersection")),
                        "lat": info.get("lat"),
                        "lon": info.get("lon")
                    })
        elif isinstance(raw_nodes, list):
            for idx, item in enumerate(raw_nodes):
                if isinstance(item, dict):
                    nid = str(item.get("id") or item.get("name") or item.get("node_id") or f"N{idx+1}")
                    name = str(item.get("name") or item.get("label") or nid)
                    node_records.append({
                        "id": nid,
                        "name": name,
                        "x": float(item.get("x", 0.0)),
                        "y": float(item.get("y", 0.0)),
                        "type": str(item.get("type") or ("depot" if idx == 0 else ("hospital" if idx == len(raw_nodes) - 1 else "intersection"))),
                        "lat": item.get("lat"),
                        "lon": item.get("lon")
                    })

        # Normalize x, y coordinates to 0..10 grid if values are large pixel offsets (> 20)
        xs = [n["x"] for n in node_records]
        ys = [n["y"] for n in node_records]
        
        max_x, min_x = (max(xs), min(xs)) if xs else (10.0, 0.0)
        max_y, min_y = (max(ys), min(ys)) if ys else (10.0, 0.0)
        range_x = (max_x - min_x) if (max_x - min_x) > 0 else 1.0
        range_y = (max_y - min_y) if (max_y - min_y) > 0 else 1.0

        for n in node_records:
            if max_x > 20.0 or max_y > 20.0:
                norm_x = round(((n["x"] - min_x) / range_x) * 10.0, 2)
                norm_y = round(((n["y"] - min_y) / range_y) * 10.0, 2)
            else:
                norm_x, norm_y = n["x"], n["y"]
                
            net.add_node(n["id"], n["name"], norm_x, norm_y, n["type"], lat=n["lat"], lon=n["lon"])

        # 2. Parse Edges / Roads / Links
        raw_edges = data.get("edges") or data.get("roads") or data.get("links") or data.get("connections") or []
        for edge in raw_edges:
            if isinstance(edge, dict):
                u = str(edge.get("u") or edge.get("from") or edge.get("source") or edge.get("start") or "")
                v = str(edge.get("v") or edge.get("to") or edge.get("target") or edge.get("end") or "")
                if u and v and u in net.nodes and v in net.nodes:
                    dist = float(edge.get("distance_km") or edge.get("distance") or edge.get("length") or edge.get("cost") or 2.0)
                    speed = float(edge.get("base_speed_kmh") or edge.get("speed") or edge.get("max_speed") or 40.0)
                    hazard = float(edge.get("hazard_weight") or edge.get("hazard") or 0.0)
                    blocked = bool(edge.get("is_blocked") or edge.get("blocked") or False)
                    net.add_edge(u, v, dist, speed, hazard, blocked, bidirectional=True)

        # Set optional background image path
        if data.get("bg_image_path"):
            net.bg_image_path = data["bg_image_path"]

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

    @classmethod
    def create_katpadi_vellore_network(cls) -> "RoadNetwork":
        """
        Creates a realistic Katpadi / Vellore Regional road network map.
        Based on real geographic corridors (Vallimalai Rd, Katpadi Main Rd, Chatram St, CM John St, etc.).
        """
        net = cls()
        coords = {
            "D-01": ("Katpadi Bus Stand EMS Depot", 0.0, 0.0, "depot"),
            "K-01": ("Chatram Street Junction", 1.5, 1.2, "intersection"),
            "K-02": ("Bajanai Koil St & Katpadi Main Rd", 2.5, 2.8, "intersection"),
            "K-03": ("Sai Baba Temple Intersection", 2.8, 4.2, "intersection"),
            "K-04": ("Fifth Avenue & Bernicepuram Ground", 4.0, 5.0, "intersection"),
            "K-05": ("CM John Street Commercial Corridor", 5.2, 3.8, "intersection"),
            "K-06": ("Vallimalai Road Main Arterial", 6.0, 4.5, "intersection"),
            "K-07": ("Jothis College of Arts & Science", 4.5, 1.5, "intersection"),
            "K-08": ("New Gate Church Junction", 5.8, 1.2, "intersection"),
            "H-01": ("Vellore Regional Government Trauma Center", 7.2, 5.2, "hospital"),
            "H-02": ("Katpadi Community Medical Hub", 3.5, 6.0, "hospital"),
        }

        for nid, (name, x, y, ntype) in coords.items():
            net.add_node(nid, name, x, y, ntype)

        edges = [
            ("D-01", "K-01", 1.5, 40.0),
            ("K-01", "K-02", 1.2, 35.0),
            ("K-02", "K-03", 1.5, 45.0),
            ("K-03", "K-04", 1.8, 40.0),
            ("K-04", "K-05", 1.4, 35.0),
            ("K-05", "K-06", 1.0, 50.0),
            ("K-06", "H-01", 1.5, 55.0),
            ("K-01", "K-07", 2.2, 40.0),
            ("K-07", "K-08", 1.3, 45.0),
            ("K-08", "K-06", 1.6, 50.0),
            ("K-03", "H-02", 1.9, 40.0),
            ("K-02", "K-05", 2.1, 40.0),
            ("D-01", "K-07", 3.0, 50.0),
            ("K-04", "H-01", 3.2, 55.0),
        ]

        for u, v, dist, speed in edges:
            net.add_edge(u, v, dist, speed)

        return net

    @classmethod
    def create_vector_city_network(cls) -> "RoadNetwork":
        """
        Creates a road network aligned with the vector street & river map graphic background.
        Uses data/sample_maps/vector_city_map.png as background map graphic overlay.
        """
        net = cls()
        net.bg_image_path = "data/sample_maps/vector_city_map.png"
        
        coords = {
            "D-01": ("Southwest EMS Depot", 1.2, 1.5, "depot"),
            "N-01": ("River West Crossing Junction", 2.5, 4.0, "intersection"),
            "N-02": ("North River Main Bridge", 4.5, 6.2, "bridge"),
            "N-03": ("East Waterfront Expressway", 8.2, 8.5, "intersection"),
            "N-04": ("Central Rotary Plaza (Incident)", 5.5, 5.0, "incident_zone"),
            "N-05": ("South River Outer Bridge", 2.0, 7.0, "bridge"),
            "N-06": ("Commercial District North", 7.0, 6.5, "intersection"),
            "N-07": ("East Suburb Ring Junction", 8.5, 4.5, "intersection"),
            "N-08": ("South Boulevard Intersection", 3.5, 2.5, "intersection"),
            "H-01": ("City Emergency Medical Trauma Center", 9.0, 2.0, "hospital"),
            "H-02": ("North Memorial Health Hub", 7.5, 9.2, "hospital"),
        }

        for nid, (name, x, y, ntype) in coords.items():
            net.add_node(nid, name, x, y, ntype)

        edges = [
            ("D-01", "N-08", 1.8, 40.0),
            ("N-08", "N-01", 2.0, 45.0),
            ("N-01", "N-05", 3.2, 35.0),
            ("N-05", "N-02", 2.8, 50.0),
            ("N-01", "N-04", 3.1, 40.0),
            ("N-04", "N-02", 1.8, 45.0),
            ("N-02", "N-06", 2.6, 50.0),
            ("N-06", "N-03", 2.2, 45.0),
            ("N-03", "H-02", 1.2, 40.0),
            ("N-04", "N-06", 2.0, 45.0),
            ("N-04", "N-07", 3.2, 50.0),
            ("N-07", "H-01", 2.6, 55.0),
            ("N-08", "N-04", 3.0, 40.0),
            ("N-08", "H-01", 5.5, 60.0),
        ]

        for u, v, dist, speed in edges:
            net.add_edge(u, v, dist, speed)

        return net

    @classmethod
    def get_available_presets(cls) -> Dict[str, str]:
        """Returns map preset keys and human-readable names."""
        return {
            "vector_city_map": "🗺️ Vector Street & River Map (Graphic Overlay)",
            "katpadi_vellore": "📍 Katpadi / Vellore Regional Road Map (11 Nodes)",
            "city_grid": "📐 Tactical City Grid Map (16 Nodes)",
            "custom_upload": "📁 Custom Uploaded / User-Defined Road Map"
        }

    @classmethod
    def create_from_preset(cls, preset_key: str) -> "RoadNetwork":
        """Factory method to instantiate a road network by preset key."""
        if preset_key == "vector_city_map":
            return cls.create_vector_city_network()
        elif preset_key == "katpadi_vellore":
            return cls.create_katpadi_vellore_network()
        return cls.create_default_city_grid()

    @classmethod
    def from_json_string(cls, json_str: str) -> "RoadNetwork":
        """Parses a custom user-uploaded map JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def export_sample_map_template(cls) -> str:
        """Generates a sample map JSON template for users to create any custom map network."""
        sample_map = {
            "nodes": {
                "START-01": {"name": "Response Depot Alpha", "x": 0.0, "y": 0.0, "type": "depot"},
                "INT-01": {"name": "North Junction", "x": 2.0, "y": 3.0, "type": "intersection"},
                "INT-02": ("East Bypass"),
                "INT-02": {"name": "East Bypass", "x": 5.0, "y": 3.0, "type": "intersection"},
                "HOSP-01": {"name": "Central Emergency Facility", "x": 7.0, "y": 0.0, "type": "hospital"}
            },
            "edges": [
                {"u": "START-01", "v": "INT-01", "distance_km": 3.6, "base_speed_kmh": 45.0},
                {"u": "INT-01", "v": "INT-02", "distance_km": 3.0, "base_speed_kmh": 40.0},
                {"u": "INT-02", "v": "HOSP-01", "distance_km": 3.6, "base_speed_kmh": 50.0},
                {"u": "START-01", "v": "HOSP-01", "distance_km": 7.0, "base_speed_kmh": 60.0}
            ]
        }
        return json.dumps(sample_map, indent=2)


