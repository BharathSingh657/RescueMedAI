"""
Perception & Disaster Image Ingestion View (Tab 1).
Handles aerial image upload, PyTorch deep learning semantic segmentation, severity estimation,
and prototype spatial mapping onto simulated road network.
"""
import streamlit as st
import os
from PIL import Image
import pandas as pd

from src.backend.vision.detector import DisasterVisionDetector
from src.backend.vision.severity import DisasterSeverityEstimator
from src.backend.routing.search import a_star_search
from src.backend.hospital.notification import HospitalNotificationGenerator
from src.frontend.components.tables import safe_render_image


def render_vision_tab():
    state = st.session_state.state

    st.subheader("Stage 1 & 2: Disaster Image Ingestion & Severity Estimation")
    st.caption("Detects structural damage, fire signatures, and road obstructions to compute a transparent severity score.")

    col1, col2 = st.columns([1.1, 1.0])

    with col1:
        st.markdown("##### 📷 Disaster Scene Analysis (Pretrained Deep Learning)")
        uploaded_img = st.file_uploader("Upload Aerial/CCTV Image for Deep Learning Analysis:", type=["jpg", "png", "jpeg"])

        if uploaded_img is not None:
            raw_img = Image.open(uploaded_img)
            if st.session_state.get("last_uploaded_file") != uploaded_img.name:
                road_conds = DisasterVisionDetector.assess_road_conditions(raw_img)
                local_dets = DisasterVisionDetector.detect_hazards(raw_img)
                state.road_conditions = road_conds
                state.detections = local_dets
                state.severity = DisasterSeverityEstimator.estimate(local_dets, road_conds)
                state.image_name = uploaded_img.name
                st.session_state["last_uploaded_file"] = uploaded_img.name
                state.log_step("VisionIngestion", f"Deep learning segmentation on: {uploaded_img.name}")
        else:
            current_img_path = state.image_path
            if current_img_path and os.path.exists(current_img_path):
                raw_img = Image.open(current_img_path)
                if not state.road_conditions:
                    state.road_conditions = DisasterVisionDetector.assess_road_conditions(raw_img)
            else:
                raw_img = Image.new("RGB", (640, 480), color=(100, 100, 100))

        overlay_img = DisasterVisionDetector.draw_detection_overlay(raw_img, state.detections)
        seg_overlay = DisasterVisionDetector.draw_segmentation_overlay(raw_img, state.road_conditions)

        view_mode = st.radio(
            "Display View:",
            ["Hazard Detection Bounding Boxes", "Semantic Segmentation Mask (PyTorch)", "Raw Camera Feed"],
            horizontal=True,
            key="view_mode_select"
        )
        if view_mode == "Hazard Detection Bounding Boxes":
            safe_render_image(overlay_img, caption=f"Hazard Detection Bounding Boxes: {state.image_name}")
        elif view_mode == "Semantic Segmentation Mask (PyTorch)":
            safe_render_image(seg_overlay, caption=f"MobileNetV3 Deep Learning Semantic Segmentation Mask (Azure: Flood, Crimson: Rubble)")
        else:
            safe_render_image(raw_img, caption=f"Raw Incident Feed: {state.image_name}")

    with col2:
        st.markdown("##### 🛣️ Evaluated Road Corridor Conditions")
        if state.road_conditions:
            rc_table = []
            for rc in state.road_conditions:
                status_symbol = "🛑" if rc.condition in ["Blocked", "Flooded"] else ("⚠️" if "Damaged" in rc.condition else "✅")
                rc_table.append({
                    "Corridor Segment": rc.segment_id.split(" (")[0],
                    "Condition": f"{status_symbol} {rc.condition}",
                    "Confidence": f"{rc.confidence:.0%}",
                    "Flood Inundation": f"{rc.flood_coverage_pct:.1f}%",
                    "Debris Blockage": f"{rc.debris_blockage_pct:.1f}%",
                    "Impact": f"{rc.suggested_hazard_weight}x delay" if rc.suggested_hazard_weight > 0 else "Normal"
                })
            st.dataframe(pd.DataFrame(rc_table), use_container_width=True, hide_index=True)
        else:
            st.info("No road corridors surveyed.")

        st.markdown("##### 🔍 Identified Spatial Observations")
        if state.detections:
            det_rows = []
            for d in state.detections:
                det_rows.append({
                    "Observation": d.label.replace("_", " ").title(),
                    "Confidence": f"{d.confidence:.0%}",
                    "Source Classification": d.source,
                    "Technical Detail": d.details
                })
            st.dataframe(pd.DataFrame(det_rows), use_container_width=True, hide_index=True)

        if state.severity:
            sev = state.severity
            st.markdown("##### ⚡ Multi-Factor Disaster Severity Assessment")
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Severity Score", f"{sev.score:.2f} / 1.0")
            sc2.metric("Severity Level", sev.label)
            sc3.metric("Blocked Roads", f"{sev.blocked_road_count}")

            st.info(f"**Severity Rationale:** {sev.rationale}")

            with st.expander("📐 View Transparent Severity Formula Breakdown"):
                st.markdown(f"""
                $$\\text{{Score}} = 0.35 \\cdot \\text{{DamagedBuildings}} + 0.25 \\cdot \\text{{BlockedRoads}} + 0.20 \\cdot \\text{{FireFactor}} + 0.20 \\cdot \\text{{Density}}$$
                - **Damaged Building Ratio:** `{sev.damaged_building_ratio}` (Contribution: `{0.35 * sev.damaged_building_ratio:.3f}`)
                - **Blocked Road Normalized:** `{min(1.0, sev.blocked_road_count / 3.0)}` (Contribution: `{0.25 * min(1.0, sev.blocked_road_count / 3.0):.3f}`)
                - **Fire Factor:** `{1.0 if sev.fire_detected else 0.0}` (Contribution: `{0.20 * (1.0 if sev.fire_detected else 0.0):.3f}`)
                - **Affected Zone Density:** `{sev.affected_zone_density}` (Contribution: `{0.20 * sev.affected_zone_density:.3f}`)
                - **Final Composite Severity:** `{sev.score:.2f}` $\\implies$ **{sev.label}**
                """)

    # Prototype Spatial Hazard Mapping Section
    st.markdown("---")
    st.markdown("#### 📍 Prototype Spatial Mapping: Detected Road Condition ➔ Simulated Map Road")
    st.markdown(r"""
    > **Operational Prototype Note:** In an active UAV/drone deployment, real-time telemetry (GPS coordinates, altitude, gimbal angle, and compass orientation) automatically georeferences detected hazards directly onto GIS road networks.  
    > In this offline prototype, **you map the detected road condition to a specific road on the simulated map**. Applying the hazard immediately updates the road graph and triggers live **A\* safe path recalculation**.
    """)

    map_col1, map_col2 = st.columns([1.2, 1.0])

    with map_col1:
        # Generate descriptive edge choices
        road_graph = st.session_state.graph
        edge_options = []
        for u in road_graph.adj:
            for v in road_graph.adj[u]:
                if u < v:
                    u_name = road_graph.nodes[u]["name"]
                    v_name = road_graph.nodes[v]["name"]
                    edge_options.append((f"{u} - {v}", f"{u} ↔ {v} ({u_name} to {v_name})"))

        selected_edge_tuple = st.selectbox(
            "1. Select Road on Simulated Map to Assign Hazard Location:",
            edge_options,
            format_func=lambda x: x[1],
            index=0
        )
        selected_edge_key = selected_edge_tuple[0]
        u_edge, v_edge = selected_edge_key.split(" - ")
        curr_edge_info = road_graph.adj[u_edge][v_edge]

        # Let user map from detected road conditions
        if state.road_conditions:
            detected_cond_labels = [
                f"{rc.segment_id.split(' ')[0]}: {rc.condition} ({rc.confidence:.0%} conf, {rc.suggested_hazard_weight}x delay)"
                for rc in state.road_conditions
            ]
            selected_rc_idx = st.selectbox(
                "2. Select Detected Condition from AI Image Analysis:",
                range(len(detected_cond_labels)),
                format_func=lambda i: detected_cond_labels[i]
            )
            chosen_rc = state.road_conditions[selected_rc_idx]
            suggested_action_idx = 0 if chosen_rc.condition in ["Blocked", "Flooded"] else (1 if chosen_rc.condition == "Severely Damaged" else (2 if chosen_rc.condition == "Partially Damaged" else 3))
        else:
            suggested_action_idx = 0

        hazard_action = st.radio(
            "3. Confirm Road Condition / Hazard Impact on Map:",
            [
                "🛑 Completely Blocked / Impassable (Structural Collapse / Deep Flood)",
                "⚠️ Severe Hazard Penalty (5.0x Travel Time Delay — Mudflow / Fire Zone)",
                "⚠️ Moderate Hazard Penalty (2.0x Travel Time Delay — Debris / Partial Submersion)",
                "✅ Clear / Reopened Roadway (Normal Speed)"
            ],
            index=suggested_action_idx
        )

    with map_col2:
        st.markdown("##### ⚡ Current Corridor Status")
        st.markdown(f"- **Segment:** `{u_edge} ↔ {v_edge}`")
        st.markdown(f"- **Physical Distance:** `{curr_edge_info['distance_km']} km` (Base Time: `{curr_edge_info['base_time_min']:.1f} min`)")
        st.markdown(f"- **Current State:** {'🛑 Blocked' if curr_edge_info['is_blocked'] else ('⚠️ Hazard Active' if curr_edge_info['hazard_weight'] > 0 else '✅ Clear')}")

        if st.button("🚀 Apply Hazard to Tactical Grid & Recalculate A* Route", type="primary", use_container_width=True):
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
            state.log_step("SpatialMapping", f"Mapped hazard to road {u_edge}-{v_edge} ({hazard_action.split(' ')[1]})")

            if state.selected_route and state.selected_route.path:
                st.success(
                    f"✅ Road `{u_edge}-{v_edge}` updated! A* safely rerouted: **{' ➔ '.join(state.selected_route.path)}** "
                    f"({state.selected_route.cost:.1f} min, {state.selected_route.distance_km:.1f} km). "
                    f"Switch to **Tab 2 (Road Graph & A* Search)** to inspect the map!"
                )
            else:
                st.error("⚠️ All navigable paths to the target hospital are currently blocked! Review road closures.")
            st.rerun()
