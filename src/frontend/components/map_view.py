"""
Matplotlib Tactical Road Graph Map rendering component.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os
from typing import List, Optional
from src.backend.routing.graph import RoadNetwork


def render_road_network_figure(
    graph: RoadNetwork,
    route_path: Optional[List[str]] = None,
    incident_node: Optional[str] = None,
    goal_node: Optional[str] = None
) -> plt.Figure:
    """
    Renders an offline tactical road network map using pure Matplotlib.
    Guarantees 1-to-1 exact rendering of every physical road edge in the active graph network.
    """
    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor="#0b0f19")
    ax.set_facecolor("#0f172a")

    # Load Background Map Image Graphic if configured
    if graph.bg_image_path and os.path.exists(graph.bg_image_path):
        bg_img = Image.open(graph.bg_image_path)
        ax.imshow(bg_img, extent=[0, 10, 0, 10], aspect="auto", alpha=0.85, zorder=0)

    # 1. Draw Edges (Strict deduplication ensuring every road in graph.adj is drawn)
    seen_drawn_edges = set()
    for u in graph.adj:
        for v, edge in graph.adj[u].items():
            edge_pair = tuple(sorted([u, v]))
            if edge_pair not in seen_drawn_edges:
                seen_drawn_edges.add(edge_pair)
                if u in graph.nodes and v in graph.nodes:
                    n1 = graph.nodes[u]
                    n2 = graph.nodes[v]
                    x_vals = [n1["x"], n2["x"]]
                    y_vals = [n1["y"], n2["y"]]

                    if edge["is_blocked"]:
                        # Blocked road (Red dashed line + Cross marker)
                        ax.plot(x_vals, y_vals, color="#ef4444", linestyle="--", linewidth=2.8, alpha=0.95, zorder=2)
                        mid_x, mid_y = (n1["x"] + n2["x"]) / 2.0, (n1["y"] + n2["y"]) / 2.0
                        ax.scatter([mid_x], [mid_y], color="#ef4444", marker="x", s=140, linewidths=3, zorder=5)
                    elif edge["hazard_weight"] > 0.0:
                        # Hazard weighted road (Orange)
                        ax.plot(x_vals, y_vals, color="#f59e0b", linestyle="-", linewidth=2.6, alpha=0.88, zorder=2)
                    else:
                        # Clear normal road (Slate gray)
                        ax.plot(x_vals, y_vals, color="#334155", linestyle="-", linewidth=2.0, alpha=0.75, zorder=1)

    # 2. Highlight Computed Route Path (if provided)
    if route_path and len(route_path) > 1:
        rx = [graph.nodes[n]["x"] for n in route_path if n in graph.nodes]
        ry = [graph.nodes[n]["y"] for n in route_path if n in graph.nodes]
        if len(rx) > 1:
            # Glowing route underlay
            ax.plot(rx, ry, color="#06b6d4", linewidth=6.5, alpha=0.45, zorder=3)
            # Route main line
            ax.plot(rx, ry, color="#38bdf8", linewidth=3.4, linestyle="-", zorder=4)

    # 3. Draw Nodes
    for nid, node in graph.nodes.items():
        x, y = node["x"], node["y"]
        ntype = node.get("type", "intersection")

        if nid == incident_node:
            # Incident Zone (Red Star)
            ax.scatter([x], [y], color="#ef4444", edgecolors="#ffffff", s=320, marker="*", linewidths=1.5, zorder=7)
            ax.text(x, y + 0.35, f"{nid} [INCIDENT]", color="#fca5a5", fontsize=9, fontweight="bold", ha="center", zorder=8)
        elif "hospital" in ntype:
            # Hospital (Cyan Square)
            ax.scatter([x], [y], color="#0284c7", edgecolors="#38bdf8", s=220, marker="s", linewidths=2.0, zorder=6)
            ax.text(x, y + 0.35, f"{nid} (ER)", color="#38bdf8", fontsize=9, fontweight="bold", ha="center", zorder=8)
        elif "depot" in ntype:
            # Depot (Purple Diamond)
            ax.scatter([x], [y], color="#8b5cf6", edgecolors="#c4b5fd", s=200, marker="D", linewidths=1.8, zorder=6)
            ax.text(x, y + 0.35, f"{nid} (EMS)", color="#c4b5fd", fontsize=8.5, fontweight="bold", ha="center", zorder=8)
        else:
            # Standard Intersections
            ax.scatter([x], [y], color="#1e293b", edgecolors="#64748b", s=100, marker="o", linewidths=1.5, zorder=5)
            ax.text(x, y + 0.28, nid, color="#94a3b8", fontsize=8, ha="center", zorder=8)

    # Custom Tactical Legend
    legend_elements = [
        mpatches.Patch(color="#38bdf8", label="Active A* Safe Route"),
        mpatches.Patch(color="#334155", label="Clear Arterial Road"),
        mpatches.Patch(color="#f59e0b", label="Hazard-Penalized Road"),
        plt.Line2D([0], [0], color="#ef4444", marker="x", linestyle="--", label="Blocked / Impassable Road", markersize=9),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#ef4444", markersize=12, label="Disaster Site"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#0284c7", markersize=10, label="Hospital Hub"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", facecolor="#131c2e", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8.5)

    ax.set_title(f"Tactical Road Graph ({len(seen_drawn_edges)} Total Corridors Rendered)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Grid Easting (km)", color="#64748b", fontsize=9)
    ax.set_ylabel("Grid Northing (km)", color="#64748b", fontsize=9)
    ax.tick_params(colors="#475569", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.grid(True, color="#1e293b", linestyle=":", alpha=0.6)

    fig.tight_layout()
    return fig
