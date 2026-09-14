"""
Tactical Road Graph & A* Search View (Tab 2).
Provides interactive tactical grid map visualization, prototype spatial mapping of AI detections onto simulated graph edges,
manual road blockage simulator, and comparative search algorithm benchmarks.
"""
import streamlit as st
import pandas as pd

from src.backend.routing.graph import RoadNetwork
from src.backend.routing.search import a_star_search
from src.backend.routing.benchmark import RoutingBenchmarkEngine
from src.backend.hospital.notification import HospitalNotificationGenerator
from src.frontend.components.map_view import render_road_network_figure


def render_routing_tab():
    state = st.session_state.state

    st.subheader("🗺️ Road Network Graph & Dynamic A* Search Rerouting")
    st.caption("A* search minimizes $f(n) = g(n) + h(n)$ with dynamic hazard weights, travel-time delays, and physical road blockages.")

    # 1. MAP PRESET SELECTOR
    presets = RoadNetwork.get_available_presets()
    preset_keys = list(presets.keys())
    selected_preset_key = st.selectbox(
        "🗺️ Active Regional Road Map Network Preset:",
        preset_keys,
        format_func=lambda k: presets[k],
        key="active_map_preset_select"
    )

    if selected_preset_key == "custom_upload":
        st.info("📁 **Custom Map Mode**: Upload a custom JSON map specification file containing nodes, coordinates, and road edges.")
        c_file = st.file_uploader("Upload Custom Map JSON Specification:", type=["json"])
        if c_file is not None:
            if st.session_state.get("last_uploaded_map_file") != c_file.name:
                try:
                    json_content = c_file.read().decode("utf-8")
                    custom_net = RoadNetwork.from_json_string(json_content)
                    st.session_state.graph = custom_net
                    st.session_state.current_map_preset = "custom_upload"
                    st.session_state["last_uploaded_map_file"] = c_file.name
                    nodes_list = list(custom_net.nodes.keys())
                    state.incident_node = nodes_list[0]
                    state.destination_node = nodes_list[-1]
                    state.selected_route = a_star_search(custom_net, state.incident_node, state.destination_node)
                    st.success(f"Loaded custom map with {len(custom_net.nodes)} nodes!")
                except Exception as e:
                    st.error(f"Invalid Map Specification: {e}")

        st.download_button(
            "📥 Download Custom Map JSON Template",
            data=RoadNetwork.export_sample_map_template(),
            file_name="sample_custom_map.json",
            mime="application/json"
        )
    elif st.session_state.get("current_map_preset") != selected_preset_key:
        st.session_state.graph = RoadNetwork.create_from_preset(selected_preset_key)
        st.session_state.current_map_preset = selected_preset_key
        nodes_list = list(st.session_state.graph.nodes.keys())
        if state.incident_node not in nodes_list:
            state.incident_node = nodes_list[1] if len(nodes_list) > 1 else nodes_list[0]
        if state.destination_node not in nodes_list:
            state.destination_node = "H-01" if "H-01" in nodes_list else nodes_list[-1]
        state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
        st.rerun()

    # 2. MAIN LAYOUT: MAP FIGURE + ACTIVE ROUTE METRICS
    col_map, col_search = st.columns([1.25, 1.0])

    with col_map:
        st.markdown("##### 🗺️ Simulated Tactical Road Network Map")
        fig = render_road_network_figure(
            graph=st.session_state.graph,
            route_path=state.selected_route.path if state.selected_route else None,
            incident_node=state.incident_node,
            goal_node=state.destination_node
        )
        st.pyplot(fig)

    with col_search:
        st.markdown("##### ⏱️ Active Optimal Path (A* Search)")
        if state.selected_route and state.selected_route.path:
            sr = state.selected_route
            r1, r2, r3 = st.columns(3)
            r1.metric("Effective Cost", f"{sr.cost:.1f} min")
            r2.metric("Total Distance", f"{sr.distance_km:.1f} km")
            r3.metric("Nodes Expanded", f"{sr.nodes_expanded}")

            path_display = " ➔ ".join([f"`{node}`" for node in sr.path])
            st.markdown(f"**Navigated Path:** {path_display}")
        else:
            st.error("⚠️ Destination completely isolated! All passable routes are blocked by disaster hazards.")

        st.markdown("---")
        st.markdown("##### 🔬 Search Algorithm Benchmark Comparison")
        st.caption("Empirical performance evaluation on active road graph:")

        bench_data = RoutingBenchmarkEngine.run_benchmark(
            st.session_state.graph, state.incident_node, state.destination_node
        )
        bench_df = pd.DataFrame(bench_data["summary_table"])
        st.dataframe(bench_df[["Algorithm", "Path Found", "Cost", "Distance (km)", "Nodes Expanded", "Runtime (ms)"]], hide_index=True, use_container_width=True)

        verif = bench_data["verification"]
        if verif["optimal_cost_matched"]:
            st.success(f"✅ **Optimality Verified:** A* and Dijkstra found identical minimal cost ({bench_data['results']['A*'].cost:.2f} min).")
        st.info(f"ℹ️ **Search Efficiency:** A* expanded `{bench_data['results']['A*'].nodes_expanded}` nodes vs. Dijkstra `{bench_data['results']['Dijkstra'].nodes_expanded}` nodes.")

    # BUILD STRICT UNIQUE EDGES LIST FOR ACTIVE MAP NETWORK
    road_graph = st.session_state.graph
    seen_edges = set()
    edge_options = []
    for u in road_graph.adj:
        for v in road_graph.adj[u]:
            edge_pair = tuple(sorted([u, v]))
            if edge_pair not in seen_edges:
                seen_edges.add(edge_pair)
                u_id, v_id = edge_pair
                u_name = road_graph.nodes[u_id]["name"]
                v_name = road_graph.nodes[v_id]["name"]
                edge_options.append((f"{u_id} - {v_id}", f"{u_id} ↔ {v_id} ({u_name} to {v_name})"))

    # 3. PROTOTYPE SPATIAL MAPPING SECTION
    st.markdown("---")
    st.markdown("#### 📍 Prototype Spatial Mapping: AI Hazard Detection ➔ Simulated Map Edge")
    st.markdown(r"""
    > **Prototype Operational Concept:** In live field deployments, drone telemetry (GPS coordinates & gimbal pitch/yaw) georeferences detected hazards directly onto GIS maps.  
    > In this simulation prototype, **you map AI-detected road conditions to a specific road segment on the simulated map**. Applying the hazard immediately updates edge weights in the graph and triggers live **A\* safe path recalculation**.
    """)

    map_col1, map_col2 = st.columns([1.2, 1.0])

    with map_col1:
        selected_edge_tuple = st.selectbox(
            f"1. Select Road Segment on Active Map (Total {len(edge_options)} Roads):",
            edge_options,
            format_func=lambda x: x[1],
            index=0,
            key="proto_edge_select"
        )
        selected_edge_key = selected_edge_tuple[0]
        u_edge, v_edge = selected_edge_key.split(" - ")
        curr_edge_info = road_graph.adj[u_edge][v_edge]

        if state.road_conditions:
            detected_cond_labels = [
                f"{rc.segment_id.split(' ')[0]}: {rc.condition} ({rc.confidence:.0%} conf, {rc.suggested_hazard_weight}x delay)"
                for rc in state.road_conditions
            ]
            selected_rc_idx = st.selectbox(
                "2. Select AI-Detected Condition from Image Analysis:",
                range(len(detected_cond_labels)),
                format_func=lambda i: detected_cond_labels[i],
                key="proto_rc_select"
            )
            chosen_rc = state.road_conditions[selected_rc_idx]
            suggested_action_idx = 0 if chosen_rc.condition in ["Blocked", "Flooded"] else (1 if chosen_rc.condition == "Severely Damaged" else (2 if chosen_rc.condition == "Partially Damaged" else 3))
        else:
            suggested_action_idx = 0

        hazard_action = st.radio(
            "3. Confirm Edge Status / Hazard Penalty on Graph:",
            [
                "🛑 Completely Blocked / Impassable (Structural Collapse / Deep Flood)",
                "⚠️ Severe Hazard Penalty (5.0x Travel Delay — Mudflow / Active Fire)",
                "⚠️ Moderate Hazard Penalty (2.0x Travel Delay — Debris / Partial Submersion)",
                "✅ Clear / Reopened Roadway (Normal Speed)"
            ],
            index=suggested_action_idx,
            key="proto_hazard_action"
        )

    with map_col2:
        st.markdown("##### ⚡ Active Edge Telemetry")
        st.markdown(f"- **Corridor:** `{u_edge} ↔ {v_edge}`")
        st.markdown(f"- **Distance:** `{curr_edge_info['distance_km']} km` (Base Time: `{curr_edge_info['base_time_min']:.1f} min`)")
        st.markdown(f"- **Current State:** {'🛑 Blocked' if curr_edge_info['is_blocked'] else ('⚠️ Hazard Active (' + str(curr_edge_info['hazard_weight']) + 'x)' if curr_edge_info['hazard_weight'] > 0 else '✅ Clear')}")

        if st.button("🚀 Apply Hazard & Recalculate A* Safe Path", type="primary", use_container_width=True, key="btn_apply_proto_hazard"):
            if "Completely Blocked" in hazard_action:
                road_graph.set_edge_status(u_edge, v_edge, is_blocked=True)
                if (u_edge, v_edge) not in state.blocked_edges and (v_edge, u_edge) not in state.blocked_edges:
                    state.blocked_edges.append((u_edge, v_edge))
            elif "Severe Hazard" in hazard_action:
                road_graph.set_edge_status(u_edge, v_edge, is_blocked=False, hazard_weight=5.0)
                state.hazard_edges[(u_edge, v_edge)] = 5.0
            elif "Moderate Hazard" in hazard_action:
                road_graph.set_edge_status(u_edge, v_edge, is_blocked=False, hazard_weight=2.0)
                state.hazard_edges[(u_edge, v_edge)] = 2.0
            else:
                road_graph.set_edge_status(u_edge, v_edge, is_blocked=False, hazard_weight=0.0)
                state.blocked_edges = [e for e in state.blocked_edges if e != (u_edge, v_edge) and e != (v_edge, u_edge)]
                state.hazard_edges.pop((u_edge, v_edge), None)
                state.hazard_edges.pop((v_edge, u_edge), None)

            # Recalculate A* route immediately
            state.selected_route = a_star_search(road_graph, state.incident_node, state.destination_node)
            triage_prio = state.triage.priority if state.triage else "RED"
            state.resource_assignment = st.session_state.allocator.allocate_priority_aware(
                state.incident_node, triage_prio, state.disaster_type
            )
            state.hospital_notification = HospitalNotificationGenerator.generate(state)
            state.log_step("SpatialMapping", f"Mapped hazard to road {u_edge}-{v_edge}")

            if state.selected_route and state.selected_route.path:
                st.success(
                    f"✅ Edge `{u_edge}-{v_edge}` updated! A* rerouted path: **{' ➔ '.join(state.selected_route.path)}** "
                    f"({state.selected_route.cost:.1f} min, {state.selected_route.distance_km:.1f} km)."
                )
            else:
                st.error("⚠️ Destination is isolated by road closures!")
            st.rerun()

    # 4. MANUAL ROAD BLOCKAGE / WHAT-IF SIMULATOR
    st.markdown("---")
    st.markdown("#### 🚧 Manual Road Blockage & What-If Rerouting Simulator")
    
    nodes_list = list(st.session_state.graph.nodes.keys())
    node_names = [f"{nid} ({st.session_state.graph.nodes[nid]['name']})" for nid in nodes_list]
    
    c_n1, c_n2 = st.columns(2)
    cur_start_idx = nodes_list.index(state.incident_node) if state.incident_node in nodes_list else 0
    cur_goal_idx = nodes_list.index(state.destination_node) if state.destination_node in nodes_list else (len(nodes_list) - 1)
    
    sel_start_idx = c_n1.selectbox("Disaster Origin Location (Start):", range(len(nodes_list)), format_func=lambda i: node_names[i], index=cur_start_idx, key="sel_start_node")
    sel_goal_idx = c_n2.selectbox("Target Medical Facility (Goal):", range(len(nodes_list)), format_func=lambda i: node_names[i], index=cur_goal_idx, key="sel_goal_node")

    new_start_node = nodes_list[sel_start_idx]
    new_goal_node = nodes_list[sel_goal_idx]
    
    if new_start_node != state.incident_node or new_goal_node != state.destination_node:
        state.incident_node = new_start_node
        state.destination_node = new_goal_node
        state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
        st.rerun()

    manual_edge_tuple = st.selectbox(
        f"Select Road Segment to Toggle Closure (Total {len(edge_options)} Roads):",
        edge_options,
        format_func=lambda x: x[1],
        key="manual_edge_toggle"
    )
    u_sel, v_sel = manual_edge_tuple[0].split(" - ")
    current_blocked = st.session_state.graph.adj[u_sel][v_sel]["is_blocked"]

    bcol1, bcol2 = st.columns(2)
    if bcol1.button("🛑 Block Selected Road Segment", disabled=current_blocked, key="btn_block_edge"):
        st.session_state.graph.set_edge_status(u_sel, v_sel, is_blocked=True)
        state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
        st.rerun()

    if bcol2.button("✅ Clear / Reopen Selected Road", disabled=not current_blocked, key="btn_clear_edge"):
        st.session_state.graph.set_edge_status(u_sel, v_sel, is_blocked=False, hazard_weight=0.0)
        state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
        st.rerun()
