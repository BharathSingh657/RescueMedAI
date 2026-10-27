"""
Tactical Road Graph & A* Search View (Tab 2).
Provides interactive tactical grid map visualization, prototype spatial mapping of AI detections onto simulated graph edges,
manual road blockage simulator, and comparative search algorithm benchmarks.
"""
import streamlit as st
import pandas as pd

import os
import math
from PIL import Image, ImageDraw
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_IMAGE_COORDS = True
except Exception:
    HAS_IMAGE_COORDS = False

from src.backend.routing.graph import RoadNetwork
from src.backend.routing.search import a_star_search
from src.backend.routing.benchmark import RoutingBenchmarkEngine
from src.backend.hospital.notification import HospitalNotificationGenerator
from src.frontend.components.map_view import render_road_network_figure


def get_next_node_id(graph, role_label: str = "") -> str:
    """Generate next available node ID based on role prefix (N-xx, H-xx, D-xx)."""
    existing = set(graph.nodes.keys()) if graph and graph.nodes else set()
    if "Hospital" in role_label:
        prefix = "H-"
    elif "Depot" in role_label:
        prefix = "D-"
    else:
        prefix = "N-"
    idx = 1
    while f"{prefix}{idx:02d}" in existing:
        idx += 1
    return f"{prefix}{idx:02d}"


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
                    if state.image_path:
                        custom_net.bg_image_path = state.image_path
                    if state.detections:
                        RoadNetwork.apply_vision_detections(custom_net, state.detections)

                    st.session_state.graph = custom_net
                    st.session_state.current_map_preset = "custom_upload"
                    st.session_state["last_uploaded_map_file"] = c_file.name
                    from src.backend.allocation.allocator import EmergencyResourceAllocator
                    st.session_state.allocator = EmergencyResourceAllocator(custom_net)

                    nodes_list = list(custom_net.nodes.keys())
                    state.incident_node = nodes_list[0]
                    state.destination_node = nodes_list[-1]
                    state.selected_route = a_star_search(custom_net, state.incident_node, state.destination_node)
                    triage_prio = state.triage.priority if state.triage else "RED"
                    state.resource_assignment = st.session_state.allocator.allocate_priority_aware(
                        state.incident_node, triage_prio, state.disaster_type
                    )
                    st.success(f"✅ Loaded custom road map '{c_file.name}' with {len(custom_net.nodes)} nodes & {sum(len(v) for v in custom_net.adj.values()) // 2} corridors!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid Map Specification: {e}")

        st.download_button(
            "📥 Download Custom Map JSON Template",
            data=RoadNetwork.export_sample_map_template(),
            file_name="sample_custom_map.json",
            mime="application/json"
        )
    elif st.session_state.get("current_map_preset") != selected_preset_key:
        new_net = RoadNetwork.create_from_preset(selected_preset_key)
        if state.image_path:
            new_net.bg_image_path = state.image_path
        if state.detections and st.session_state.get("is_custom_image_uploaded"):
            RoadNetwork.apply_vision_detections(new_net, state.detections)

        st.session_state.graph = new_net
        st.session_state.current_map_preset = selected_preset_key
        from src.backend.allocation.allocator import EmergencyResourceAllocator
        st.session_state.allocator = EmergencyResourceAllocator(new_net)

        nodes_list = list(st.session_state.graph.nodes.keys())
        if state.incident_node not in nodes_list:
            state.incident_node = nodes_list[1] if len(nodes_list) > 1 else nodes_list[0]
        if state.destination_node not in nodes_list:
            state.destination_node = "H-01" if "H-01" in nodes_list else nodes_list[-1]
        state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
        triage_prio = state.triage.priority if state.triage else "RED"
        state.resource_assignment = st.session_state.allocator.allocate_priority_aware(
            state.incident_node, triage_prio, state.disaster_type
        )
        st.rerun()

    # 2. MAIN LAYOUT: MAP FIGURE + ACTIVE ROUTE METRICS
    col_map, col_search = st.columns([1.25, 1.0])

    with col_map:
        mode_options = []
        if HAS_IMAGE_COORDS:
            mode_options.append("🎯 Cursor Pin Pick & Drop Mode")
        mode_options.append("📐 Tactical Network View (Static Plot)")

        map_mode = st.radio(
            "Display Mode:",
            mode_options,
            horizontal=True,
            key="radio_tactical_map_mode"
        )

        if map_mode == "🎯 Cursor Pin Pick & Drop Mode":
            picked_pin = st.session_state.get("picked_pin")
            if picked_pin and picked_pin not in st.session_state.graph.nodes:
                picked_pin = None
                st.session_state["picked_pin"] = None

            # Tactical Cursor Action Bar
            action_mode = st.radio(
                "Cursor Action Tool:",
                ["📍 Move Existing Pins (Pick & Drop)", "➕ Create & Place New Node"],
                horizontal=True,
                key="cursor_action_tool_radio"
            )

            if action_mode == "➕ Create & Place New Node":
                picked_pin = None
                create_role = st.selectbox(
                    "New Node Role:",
                    ["⚪ Road Intersection / Waypoint", "⭐️ Incident Site", "🏥 Field Hospital", "🚑 Forward Ambulance Depot"],
                    key="create_role_choice"
                )
                sug_nid = get_next_node_id(st.session_state.graph, create_role)
                c_c1, c_c2 = st.columns([1, 1.5])
                with c_c1:
                    create_nid_val = st.text_input("Node ID:", value=sug_nid, key="create_nid_input_val")
                with c_c2:
                    create_conn = st.selectbox(
                        "Auto-Connect Roads:",
                        ["Auto-Connect to 2 Nearest Nodes", "Auto-Connect to 1 Nearest Node", "Isolated (No Links)"],
                        key="create_conn_choice"
                    )
                st.warning(
                    f"👉 **Click ANYWHERE on the aerial photo below** with your cursor to place node `{create_nid_val}`! Roads will auto-connect."
                )

            else:
                if picked_pin:
                    p_node = st.session_state.graph.nodes[picked_pin]
                    st.success(
                        f"🎯 **PIN IN HAND: `{picked_pin}` ({p_node['name']})**  \n"
                        f"👉 **Click ANYWHERE on the map/aerial photo below** with your cursor to DROP this pin! Road connections and A* route will instantly re-route."
                    )
                else:
                    st.info(
                        "👆 **Direct Cursor Pick & Drop:** Click any pin on the map to pick it up, or tap a Quick-Pick button below, then click where you want to drop it!"
                    )

                # Quick-Pick Toolbar (4 clean, spacious columns)
                qp_c1, qp_c2, qp_c3, qp_c4 = st.columns([1.1, 1.1, 1.1, 1.4])
                with qp_c1:
                    is_inc = (picked_pin == state.incident_node)
                    if st.button(f"{'🎯 ' if is_inc else '⭐️ '}Incident ({state.incident_node})", key="btn_pick_inc", type="primary" if is_inc else "secondary", use_container_width=True):
                        st.session_state["picked_pin"] = None if is_inc else state.incident_node
                        st.rerun()

                with qp_c2:
                    is_hosp = (picked_pin == state.destination_node)
                    if st.button(f"{'🎯 ' if is_hosp else '🏥 '}Hospital ({state.destination_node})", key="btn_pick_hosp", type="primary" if is_hosp else "secondary", use_container_width=True):
                        st.session_state["picked_pin"] = None if is_hosp else state.destination_node
                        st.rerun()

                with qp_c3:
                    depots = [nid for nid, n in st.session_state.graph.nodes.items() if "depot" in n.get("type", "")]
                    depot_nid = depots[0] if depots else None
                    if depot_nid:
                        is_dep = (picked_pin == depot_nid)
                        if st.button(f"{'🎯 ' if is_dep else '🚑 '}Depot ({depot_nid})", key="btn_pick_depot", type="primary" if is_dep else "secondary", use_container_width=True):
                            st.session_state["picked_pin"] = None if is_dep else depot_nid
                            st.rerun()

                with qp_c4:
                    all_nids = list(st.session_state.graph.nodes.keys())
                    sel_opt = st.selectbox(
                        "Pick Other Pin:",
                        ["⚪ Pick Other..."] + all_nids,
                        key="sel_pick_node_drop",
                        label_visibility="collapsed"
                    )
                    if sel_opt != "⚪ Pick Other...":
                        st.session_state["picked_pin"] = sel_opt
                        st.rerun()

            # Load and annotate aerial photo or tactical canvas
            img_src = state.image_path if state.image_path and os.path.exists(state.image_path) else st.session_state.graph.bg_image_path
            if img_src and os.path.exists(img_src):
                base_img = Image.open(img_src).convert("RGB")
                tint = Image.new("RGB", base_img.size, (15, 23, 42))
                base_img = Image.blend(base_img, tint, 0.22)
            else:
                base_img = Image.new("RGB", (720, 520), color=(15, 23, 42))
                grid_draw = ImageDraw.Draw(base_img)
                for gx in range(0, 720, 60):
                    grid_draw.line([(gx, 0), (gx, 520)], fill=(30, 41, 59), width=1)
                for gy in range(0, 520, 60):
                    grid_draw.line([(0, gy), (720, gy)], fill=(30, 41, 59), width=1)

            disp_w = 720
            disp_h = max(380, int(disp_w * (base_img.height / float(base_img.width))))
            canvas_img = base_img.resize((disp_w, disp_h), Image.BILINEAR)
            draw = ImageDraw.Draw(canvas_img)

            # Spatial mapping bounds (stable at minimum 0-10)
            xs = [n["x"] for n in st.session_state.graph.nodes.values()] if st.session_state.graph.nodes else [0.0, 10.0]
            ys = [n["y"] for n in st.session_state.graph.nodes.values()] if st.session_state.graph.nodes else [0.0, 10.0]
            min_x = min(0.0, min(xs))
            max_x = max(10.0, max(xs))
            min_y = min(0.0, min(ys))
            max_y = max(10.0, max(ys))
            span_x = (max_x - min_x) if (max_x - min_x) > 0 else 10.0
            span_y = (max_y - min_y) if (max_y - min_y) > 0 else 10.0

            def to_px(gx, gy):
                px = int(((gx - min_x) / span_x) * disp_w)
                py = int(((max_y - gy) / span_y) * disp_h)
                return px, py

            # Draw road edges
            for u in st.session_state.graph.adj:
                for v, e in st.session_state.graph.adj[u].items():
                    if u < v and u in st.session_state.graph.nodes and v in st.session_state.graph.nodes:
                        x1, y1 = to_px(st.session_state.graph.nodes[u]["x"], st.session_state.graph.nodes[u]["y"])
                        x2, y2 = to_px(st.session_state.graph.nodes[v]["x"], st.session_state.graph.nodes[v]["y"])
                        if e["is_blocked"]:
                            draw.line([x1, y1, x2, y2], fill=(239, 68, 68), width=3)
                            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                            draw.line([mx - 5, my - 5, mx + 5, my + 5], fill=(239, 68, 68), width=2)
                            draw.line([mx - 5, my + 5, mx + 5, my - 5], fill=(239, 68, 68), width=2)
                        elif e["hazard_weight"] > 0:
                            draw.line([x1, y1, x2, y2], fill=(245, 158, 11), width=2)
                        else:
                            draw.line([x1, y1, x2, y2], fill=(148, 163, 184), width=2)

            # Draw active route path
            if state.selected_route and len(state.selected_route.path) > 1:
                r_pts = [to_px(st.session_state.graph.nodes[nid]["x"], st.session_state.graph.nodes[nid]["y"]) for nid in state.selected_route.path if nid in st.session_state.graph.nodes]
                if len(r_pts) > 1:
                    for i in range(len(r_pts) - 1):
                        draw.line([r_pts[i], r_pts[i+1]], fill=(56, 189, 248), width=6)
                        draw.line([r_pts[i], r_pts[i+1]], fill=(255, 255, 255), width=2)

            # Draw pins
            for nid, node in st.session_state.graph.nodes.items():
                px, py = to_px(node["x"], node["y"])
                is_in_hand = (nid == picked_pin)
                rad = 14 if is_in_hand else 10

                if is_in_hand:
                    draw.ellipse([px - 22, py - 22, px + 22, py + 22], outline=(250, 204, 21), width=3)
                    draw.line([px - 28, py, px + 28, py], fill=(250, 204, 21), width=2)
                    draw.line([px, py - 28, px, py + 28], fill=(250, 204, 21), width=2)
                    fill_color = (250, 204, 21)
                    outline_color = (255, 255, 255)
                elif nid == state.incident_node:
                    fill_color = (239, 68, 68)
                    outline_color = (255, 255, 255)
                elif "hospital" in node.get("type", "") or nid == state.destination_node:
                    fill_color = (2, 132, 199)
                    outline_color = (255, 255, 255)
                elif "depot" in node.get("type", ""):
                    fill_color = (139, 92, 246)
                    outline_color = (255, 255, 255)
                else:
                    fill_color = (30, 41, 59)
                    outline_color = (203, 213, 225)

                draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=fill_color, outline=outline_color, width=2)
                label_text = f"📍 {nid} [IN HAND]" if is_in_hand else (f"⭐️ {nid}" if nid == state.incident_node else (f"🏥 {nid}" if 'hospital' in node.get('type','') or nid == state.destination_node else (f"🚑 {nid}" if 'depot' in node.get('type','') else nid)))
                draw.rectangle([px + rad + 2, py - 8, px + rad + 6 + len(label_text) * 7, py + 8], fill=(15, 23, 42), outline=outline_color if is_in_hand else None)
                draw.text((px + rad + 4, py - 6), label_text, fill=(250, 204, 21) if is_in_hand else (255, 255, 255))

            # Streamlit interactive image coordinates
            click_pos = streamlit_image_coordinates(canvas_img, key="interactive_cursor_pin_canvas")

            if click_pos is not None:
                last_click = st.session_state.get("last_interactive_cursor_click")
                if click_pos != last_click:
                    st.session_state["last_interactive_cursor_click"] = click_pos
                    cx, cy = click_pos["x"], click_pos["y"]

                    # 1. CREATE MODE ACTION
                    if action_mode == "➕ Create & Place New Node":
                        new_gx = round(min_x + (cx / float(disp_w)) * span_x, 2)
                        new_gy = round(max_y - (cy / float(disp_h)) * span_y, 2)
                        new_gx = max(min_x, min(max_x, new_gx))
                        new_gy = max(min_y, min(max_y, new_gy))

                        c_role = st.session_state.get("create_role_choice", "⚪ Road Intersection / Waypoint")
                        c_nid = st.session_state.get("create_nid_input_val", "").strip().upper()
                        if not c_nid or c_nid in st.session_state.graph.nodes:
                            c_nid = get_next_node_id(st.session_state.graph, c_role)

                        role_str = "hospital" if "Hospital" in c_role else ("depot" if "Depot" in c_role else "intersection")
                        st.session_state.graph.add_node(c_nid, f"Node {c_nid}", new_gx, new_gy, node_type=role_str)

                        conn_rule = st.session_state.get("create_conn_choice", "Auto-Connect to 2 Nearest Nodes")
                        if "Nearest" in conn_rule:
                            k = 2 if "2" in conn_rule else 1
                            cands = []
                            for nid in st.session_state.graph.nodes:
                                if nid != c_nid:
                                    d = st.session_state.graph.euclidean_distance(c_nid, nid)
                                    cands.append((d, nid))
                            cands.sort()
                            for d, nid in cands[:k]:
                                st.session_state.graph.add_edge(c_nid, nid, distance_km=round(d, 2), base_speed_kmh=45.0, bidirectional=True)

                        if "Incident" in c_role:
                            state.incident_node = c_nid
                        elif "Hospital" in c_role and len(st.session_state.graph.nodes) <= 3:
                            state.destination_node = c_nid

                        state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
                        HospitalNotificationGenerator.generate(state)
                        st.toast(f"✨ Created node {c_nid} at ({new_gx}, {new_gy}) km! Road network updated.")
                        st.rerun()

                    # 2. PICK & DROP ACTION
                    closest_nid = None
                    min_dist_px = 999999.0
                    for nid, node in st.session_state.graph.nodes.items():
                        nx, ny = to_px(node["x"], node["y"])
                        d = math.hypot(cx - nx, cy - ny)
                        if d < min_dist_px:
                            min_dist_px = d
                            closest_nid = nid

                    if min_dist_px <= 28:
                        if picked_pin == closest_nid:
                            st.session_state["picked_pin"] = None
                            st.toast(f"Released pin {closest_nid}.")
                        else:
                            st.session_state["picked_pin"] = closest_nid
                            st.toast(f"🎯 Picked up pin {closest_nid}! Click anywhere on the map to DROP it.")
                        st.rerun()
                    else:
                        if picked_pin and picked_pin in st.session_state.graph.nodes:
                            new_gx = round(min_x + (cx / float(disp_w)) * span_x, 2)
                            new_gy = round(max_y - (cy / float(disp_h)) * span_y, 2)
                            new_gx = max(min_x, min(max_x, new_gx))
                            new_gy = max(min_y, min(max_y, new_gy))

                            st.session_state.graph.nodes[picked_pin]["x"] = new_gx
                            st.session_state.graph.nodes[picked_pin]["y"] = new_gy

                            for nbr in st.session_state.graph.adj.get(picked_pin, {}):
                                dist = st.session_state.graph.euclidean_distance(picked_pin, nbr)
                                st.session_state.graph.adj[picked_pin][nbr]["distance_km"] = dist
                                spd = st.session_state.graph.adj[picked_pin][nbr].get("base_speed_kmh", 45.0)
                                st.session_state.graph.adj[picked_pin][nbr]["base_time_min"] = (dist / spd) * 60.0
                                if picked_pin in st.session_state.graph.adj.get(nbr, {}):
                                    st.session_state.graph.adj[nbr][picked_pin]["distance_km"] = dist
                                    st.session_state.graph.adj[nbr][picked_pin]["base_time_min"] = st.session_state.graph.adj[picked_pin][nbr]["base_time_min"]

                            state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
                            HospitalNotificationGenerator.generate(state)

                            st.session_state["picked_pin"] = None
                            st.toast(f"📍 Successfully dropped {picked_pin} at ({new_gx}, {new_gy}) km! Route recalculated.")
                            st.rerun()
                        elif min_dist_px <= 45:
                            st.session_state["picked_pin"] = closest_nid
                            st.toast(f"🎯 Picked up pin {closest_nid}! Click anywhere on the map to DROP it.")
                            st.rerun()

            col_btn_reset, col_del = st.columns([1, 1.2])
            with col_btn_reset:
                if st.button("🔄 Reset Pin Positions to Original Map", key="btn_reset_pins_canvas", use_container_width=True):
                    preset = st.session_state.get("current_map_preset", "city_grid")
                    st.session_state.graph = RoadNetwork.create_from_preset(preset)
                    if state.image_path:
                        st.session_state.graph.bg_image_path = state.image_path
                    if state.detections:
                        RoadNetwork.apply_vision_detections(st.session_state.graph, state.detections)
                    state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
                    st.session_state["picked_pin"] = None
                    st.rerun()

            with col_del:
                deletable = [nid for nid in st.session_state.graph.nodes if nid not in [state.incident_node, state.destination_node]]
                if deletable:
                    d_c1, d_c2 = st.columns([2, 1])
                    with d_c1:
                        del_target = st.selectbox("Delete Node:", deletable, key="sel_del_target_node", label_visibility="collapsed")
                    with d_c2:
                        if st.button(f"🗑️ Delete", key="btn_del_target_node", use_container_width=True):
                            st.session_state.graph.nodes.pop(del_target, None)
                            st.session_state.graph.adj.pop(del_target, None)
                            for u in list(st.session_state.graph.adj.keys()):
                                st.session_state.graph.adj[u].pop(del_target, None)
                            state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
                            st.toast(f"🗑️ Deleted node {del_target} and removed associated road links.")
                            st.rerun()

        else:
            show_photo_overlay = st.toggle("🛰️ Overlay on Aerial Disaster Photo", value=True, key="toggle_aerial_bg")
            fig = render_road_network_figure(
                graph=st.session_state.graph,
                route_path=state.selected_route.path if state.selected_route else None,
                incident_node=state.incident_node,
                goal_node=state.destination_node,
                show_image_overlay=show_photo_overlay
            )
            st.pyplot(fig)

        with st.expander("📍 Calibrate Road Nodes on Photo (Adjust Pin Locations)", expanded=False):
            st.caption("Drag pins out of water, onto roads, or onto specific landmarks visible in your photo.")
            cal_c1, cal_c2, cal_c3 = st.columns([1.3, 1.0, 1.0])
            with cal_c1:
                cal_nid = st.selectbox(
                    "Pin to Reposition:",
                    list(st.session_state.graph.nodes.keys()),
                    format_func=lambda nid: f"{nid} ({st.session_state.graph.nodes[nid]['name']})",
                    key="cal_node_sel"
                )
            target_n = st.session_state.graph.nodes[cal_nid]
            with cal_c2:
                cal_x = st.slider("East-West (km):", 0.0, 10.0, float(target_n["x"]), 0.2, key=f"cx_{cal_nid}")
            with cal_c3:
                cal_y = st.slider("North-South (km):", 0.0, 10.0, float(target_n["y"]), 0.2, key=f"cy_{cal_nid}")

            if cal_x != target_n["x"] or cal_y != target_n["y"]:
                target_n["x"] = cal_x
                target_n["y"] = cal_y
                for nbr in st.session_state.graph.adj.get(cal_nid, {}):
                    dist = st.session_state.graph.euclidean_distance(cal_nid, nbr)
                    st.session_state.graph.adj[cal_nid][nbr]["distance_km"] = dist
                    st.session_state.graph.adj[cal_nid][nbr]["base_time_min"] = (dist / st.session_state.graph.adj[cal_nid][nbr]["base_speed_kmh"]) * 60.0
                    if cal_nid in st.session_state.graph.adj.get(nbr, {}):
                        st.session_state.graph.adj[nbr][cal_nid]["distance_km"] = dist
                        st.session_state.graph.adj[nbr][cal_nid]["base_time_min"] = st.session_state.graph.adj[cal_nid][nbr]["base_time_min"]
                state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
                st.rerun()

            if st.button("🔄 Reset Pin Positions to Default", key="btn_reset_pins"):
                preset = st.session_state.get("current_map_preset", "city_grid")
                st.session_state.graph = RoadNetwork.create_from_preset(preset)
                if state.image_path:
                    st.session_state.graph.bg_image_path = state.image_path
                if state.detections:
                    RoadNetwork.apply_vision_detections(st.session_state.graph, state.detections)
                state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
                st.rerun()

    with col_search:
        st.markdown("##### ⏱️ Active Optimal Path (A* Search)")
        if state.selected_route and state.selected_route.path:
            sr = state.selected_route
            if "Emergency Clearance" in sr.algorithm:
                st.warning(f"⚠️ **Emergency Heavy Clearance Path Active:** Standard corridors obstructed. Escort path requires clearing {sr.blocked_roads_avoided} blocked corridor (+{sr.blocked_roads_avoided * 20}m excavation delay).")
            else:
                st.success("✅ **Active A* Optimal Safe Route Found**")

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
