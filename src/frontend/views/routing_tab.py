"""
Tactical Road Graph & A* Search View (Tab 2).
Provides interactive grid map visualization, edge closure toggles,
and comparative search algorithm benchmarks.
"""
import streamlit as st
import pandas as pd

from src.backend.routing.graph import RoadNetwork
from src.backend.routing.search import a_star_search
from src.backend.routing.benchmark import RoutingBenchmarkEngine
from src.frontend.components.map_view import render_road_network_figure


def render_routing_tab():
    state = st.session_state.state

    st.subheader("Stage 3: Dynamic Hazard-Weighted Road Graph & A* Route Planning")
    st.caption("A* algorithm evaluates f(n) = g(n) + h(n) with dynamic travel-time penalties and impassable closures.")

    # Map Preset Selector (For Map Developer & Prototype Simulation)
    presets = RoadNetwork.get_available_presets()
    preset_keys = list(presets.keys())
    selected_preset_key = st.selectbox(
        "🗺️ Select Active Regional Road Map Network:",
        preset_keys,
        format_func=lambda k: presets[k],
        key="active_map_preset_select"
    )

    if selected_preset_key == "custom_upload":
        st.info("📁 **Custom Road Map Mode**: Upload any JSON map specification file containing custom nodes, intersections, and road coordinates.")
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
        # Update incident & goal node if needed
        nodes_list = list(st.session_state.graph.nodes.keys())
        if state.incident_node not in nodes_list:
            state.incident_node = nodes_list[1] if len(nodes_list) > 1 else nodes_list[0]
        if state.destination_node not in nodes_list:
            state.destination_node = "H-01" if "H-01" in nodes_list else nodes_list[-1]
        state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
        st.rerun()

    col_map, col_search = st.columns([1.25, 1.0])

    with col_map:
        st.markdown("##### 🗺️ Tactical Grid Map")
        fig = render_road_network_figure(
            graph=st.session_state.graph,
            route_path=state.selected_route.path if state.selected_route else None,
            incident_node=state.incident_node,
            goal_node=state.destination_node
        )
        st.pyplot(fig)

        # Dynamic Road Modification Controls
        with st.expander("🚧 Interactive Road Blockage / Hazard Rerouting Simulator"):
            st.markdown("##### 📍 Incident Origin & Destination Selection")
            nodes_list = list(st.session_state.graph.nodes.keys())
            node_names = [f"{nid} ({st.session_state.graph.nodes[nid]['name']})" for nid in nodes_list]
            
            c_n1, c_n2 = st.columns(2)
            cur_start_idx = nodes_list.index(state.incident_node) if state.incident_node in nodes_list else 0
            cur_goal_idx = nodes_list.index(state.destination_node) if state.destination_node in nodes_list else (len(nodes_list) - 1)
            
            sel_start_idx = c_n1.selectbox("Disaster Incident Location (Start):", range(len(nodes_list)), format_func=lambda i: node_names[i], index=cur_start_idx, key="sel_start_node")
            sel_goal_idx = c_n2.selectbox("Target Emergency Facility (Goal):", range(len(nodes_list)), format_func=lambda i: node_names[i], index=cur_goal_idx, key="sel_goal_node")

            new_start_node = nodes_list[sel_start_idx]
            new_goal_node = nodes_list[sel_goal_idx]
            
            if new_start_node != state.incident_node or new_goal_node != state.destination_node:
                state.incident_node = new_start_node
                state.destination_node = new_goal_node
                state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
                st.rerun()

            st.markdown("---")
            st.markdown("##### 🛑 Road Blockage / Closure Simulator")
            edge_list = [
                f"{u} - {v}"
                for u in st.session_state.graph.adj
                for v in st.session_state.graph.adj[u]
                if u < v
            ]
            selected_edge_str = st.selectbox("Select Road Segment to Toggle Closure:", edge_list)
            u_sel, v_sel = selected_edge_str.split(" - ")
            current_blocked = st.session_state.graph.adj[u_sel][v_sel]["is_blocked"]

            bcol1, bcol2 = st.columns(2)
            if bcol1.button("🛑 Block Selected Road", disabled=current_blocked):
                st.session_state.graph.set_edge_status(u_sel, v_sel, is_blocked=True)
                state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
                st.rerun()

            if bcol2.button("✅ Clear / Open Road", disabled=not current_blocked):
                st.session_state.graph.set_edge_status(u_sel, v_sel, is_blocked=False, hazard_weight=0.0)
                state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
                st.rerun()

    with col_search:
        st.markdown("##### ⏱️ Active Optimal Route")
        if state.selected_route and state.selected_route.path:
            sr = state.selected_route
            r1, r2, r3 = st.columns(3)
            r1.metric("Effective Cost", f"{sr.cost:.1f} min")
            r2.metric("Total Distance", f"{sr.distance_km:.1f} km")
            r3.metric("Nodes Expanded", f"{sr.nodes_expanded}")

            path_display = " ➔ ".join([f"`{node}`" for node in sr.path])
            st.markdown(f"**Navigated Path:** {path_display}")
        else:
            st.error("No passable route found: Destination is completely isolated by road closures!")

        st.markdown("---")
        st.markdown("##### 🔬 Comparative Search Benchmark (Experimental Scenario B)")
        st.caption("Empirical execution across identical graph conditions:")

        # Run benchmark
        bench_data = RoutingBenchmarkEngine.run_benchmark(
            st.session_state.graph, state.incident_node, state.destination_node
        )
        bench_df = pd.DataFrame(bench_data["summary_table"])
        st.dataframe(bench_df[["Algorithm", "Path Found", "Cost", "Distance (km)", "Nodes Expanded", "Runtime (ms)"]], hide_index=True, use_container_width=True)

        verif = bench_data["verification"]
        if verif["optimal_cost_matched"]:
            st.success(f"✅ **Optimality Confirmed:** A* and Dijkstra reached identical cost ({bench_data['results']['A*'].cost:.2f} min).")
        st.info(f"ℹ️ **Empirical Node Expansion Observation:** A* expanded `{bench_data['results']['A*'].nodes_expanded}` nodes vs Dijkstra `{bench_data['results']['Dijkstra'].nodes_expanded}` nodes.")
