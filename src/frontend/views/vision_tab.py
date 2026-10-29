"""
Perception & Disaster Image Ingestion View (Tab 1).
Handles aerial image upload, PyTorch deep learning semantic segmentation, hazard detection,
and transparent multi-factor disaster severity estimation.
"""
import streamlit as st
import os
from PIL import Image
import pandas as pd

from src.backend.vision.detector import DisasterVisionDetector
from src.backend.vision.severity import DisasterSeverityEstimator
from src.backend.state import SIMULATED_DATA_NOTICE
from src.frontend.components.header import render_simulation_notice
from src.frontend.components.tables import safe_render_image


def render_vision_tab():
    state = st.session_state.state

    st.subheader("🛰️ Perception & Disaster Severity Assessment")
    st.caption("PyTorch Deep Learning models perform automated hazard detection, road obstruction segmentation, and multi-factor severity scoring.")

    render_simulation_notice("Computer Vision analysis utilizes pretrained MobileNetV3 / ResNet architectures. Aerial imagery and detected bounding boxes serve as decision support for human commanders.")

    col1, col2 = st.columns([1.15, 1.0])

    with col1:
        st.markdown("##### 📷 Disaster Scene Vision Ingestion (PyTorch Deep Learning)")
        uploaded_img = st.file_uploader("Upload Aerial / Drone / CCTV Image for Deep Learning Analysis:", type=["jpg", "png", "jpeg"])

        if uploaded_img is not None:
            raw_img = Image.open(uploaded_img)
            if st.session_state.get("last_uploaded_file") != uploaded_img.name:
                custom_save_path = os.path.join("data", "sample_images", "custom_latest.jpg")
                try:
                    raw_img.convert("RGB").save(custom_save_path)
                    state.image_path = custom_save_path
                except Exception:
                    state.image_path = None

                road_conds = DisasterVisionDetector.assess_road_conditions(raw_img)
                local_dets = DisasterVisionDetector.detect_hazards(raw_img)
                state.road_conditions = road_conds
                state.detections = local_dets
                state.severity = DisasterSeverityEstimator.estimate(local_dets, road_conds)
                state.image_name = uploaded_img.name

                # Connect AI Vision to Road Graph when disaster image overlay mode is active:
                if state.image_path and "graph" in st.session_state:
                    current_preset = st.session_state.get("current_map_preset", "city_grid")
                    if current_preset in ["disaster_image_overlay", "uploaded_image_map"]:
                        st.session_state.graph.bg_image_path = state.image_path
                        from src.backend.routing.graph import RoadNetwork
                        if hasattr(st.session_state.graph, "apply_vision_detections"):
                            impacted = st.session_state.graph.apply_vision_detections(
                                local_dets, image_size=raw_img.size, clear_previous=False
                            )
                        else:
                            impacted = RoadNetwork.apply_vision_detections(
                                st.session_state.graph, local_dets, image_size=raw_img.size, clear_previous=False
                            )
                        st.session_state["vision_impacted_edges"] = impacted

                    # Recalculate A* route, resource allocation, and hospital dispatch
                    from src.backend.routing.search import a_star_search
                    from src.backend.hospital.notification import HospitalNotificationGenerator
                    state.selected_route = a_star_search(st.session_state.graph, state.incident_node, state.destination_node)
                    triage_prio = state.triage.priority if state.triage else "RED"
                    state.resource_assignment = st.session_state.allocator.allocate_priority_aware(
                        state.incident_node, triage_prio, state.disaster_type
                    )
                    state.hospital_notification = HospitalNotificationGenerator.generate(state)

                st.session_state["last_uploaded_file"] = uploaded_img.name
                st.session_state["is_custom_image_uploaded"] = True
                state.log_step("VisionIngestion", f"Deep learning segmentation on: {uploaded_img.name}")
                st.rerun()
        else:
            current_img_path = state.image_path
            if current_img_path and os.path.exists(current_img_path):
                raw_img = Image.open(current_img_path)
                if not state.road_conditions:
                    state.road_conditions = DisasterVisionDetector.assess_road_conditions(raw_img)
            else:
                raw_img = Image.new("RGB", (640, 480), color=(100, 100, 100))

        if st.session_state.get("is_custom_image_uploaded"):
            st.info(f"📷 **Custom Image Active:** `{st.session_state.get('last_uploaded_file')}`")
            impacted = st.session_state.get("vision_impacted_edges", [])
            if impacted:
                b_edges = [f"`{e['u']}↔{e['v']}` ({e['cause']})" for e in impacted if e['status'] == 'blocked']
                h_edges = [f"`{e['u']}↔{e['v']}`" for e in impacted if e['status'] != 'blocked']
                b_str = ", ".join(b_edges) if b_edges else "None"
                h_str = ", ".join(h_edges) if h_edges else "None"
                st.success(f"🛰️ **AI Vision ➔ Road Graph Bridge Active:**\n\n- **Blocked Corridors:** {b_str}\n- **Hazard-Delayed:** {h_str}\n\n👉 *Open **Tab 2 (Road Network Graph & Dynamic Routing)** to pick & drop pins directly on this aerial photo with your cursor!*")
            if st.button("🔄 Restore Predefined Scenarios", key="btn_restore_presets"):
                st.session_state["last_uploaded_file"] = None
                st.session_state["is_custom_image_uploaded"] = False
                st.session_state["vision_impacted_edges"] = []
                if st.session_state.get("scenarios"):
                    from src.frontend.main import apply_scenario
                    apply_scenario(st.session_state.scenarios[st.session_state.get("current_scenario_idx", 0)])
                st.rerun()

        overlay_img = DisasterVisionDetector.draw_detection_overlay(raw_img, state.detections)
        seg_overlay = DisasterVisionDetector.draw_segmentation_overlay(raw_img, state.road_conditions)

        view_mode = st.radio(
            "Visual Layer Mode:",
            ["Hazard Detection Bounding Boxes", "Semantic Segmentation Mask (PyTorch)", "Raw Camera Feed"],
            horizontal=True,
            key="view_mode_select"
        )
        if view_mode == "Hazard Detection Bounding Boxes":
            safe_render_image(overlay_img, caption=f"Hazard Bounding Boxes & Object Classifications: {state.image_name}")
        elif view_mode == "Semantic Segmentation Mask (PyTorch)":
            safe_render_image(seg_overlay, caption=f"MobileNetV3 Deep Learning Semantic Segmentation (Azure: Flood, Crimson: Rubble)")
        else:
            safe_render_image(raw_img, caption=f"Raw Tactical Aerial Feed: {state.image_name}")

    with col2:
        st.markdown("##### 🛣️ AI-Assessed Road Corridor Conditions")
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
                    "Impact Delay": f"{rc.suggested_hazard_weight}x" if rc.suggested_hazard_weight > 0 else "Normal"
                })
            st.dataframe(pd.DataFrame(rc_table), use_container_width=True, hide_index=True)
        else:
            st.info("No road corridors surveyed.")

        st.markdown("##### 🔍 Identified Spatial Bounding Box Detections")
        if state.detections:
            det_rows = []
            for d in state.detections:
                det_rows.append({
                    "Detected Hazard": d.label.replace("_", " ").title(),
                    "Confidence": f"{d.confidence:.0%}",
                    "Classification Source": d.source,
                    "Technical Detail": d.details
                })
            st.dataframe(pd.DataFrame(det_rows), use_container_width=True, hide_index=True)

        if state.severity:
            sev = state.severity
            st.markdown("##### ⚡ Multi-Factor Disaster Severity Index")
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Severity Score", f"{sev.score:.2f} / 1.0")
            sc2.metric("Severity Category", sev.label)
            sc3.metric("Blocked Road Segments", f"{sev.blocked_road_count}")

            st.info(f"**AI Severity Rationale:** {sev.rationale}")

            with st.expander("📐 Transparent Severity Formula Calculation"):
                st.markdown(f"""
                $$\\text{{Severity Score}} = 0.35 \\cdot \\text{{DamagedBuildings}} + 0.25 \\cdot \\text{{BlockedRoads}} + 0.20 \\cdot \\text{{FireFactor}} + 0.20 \\cdot \\text{{Density}}$$
                - **Damaged Building Ratio:** `{sev.damaged_building_ratio}` (Weight: 0.35 $\\rightarrow$ Contribution: `{0.35 * sev.damaged_building_ratio:.3f}`)
                - **Blocked Road Ratio:** `{min(1.0, sev.blocked_road_count / 3.0)}` (Weight: 0.25 $\\rightarrow$ Contribution: `{0.25 * min(1.0, sev.blocked_road_count / 3.0):.3f}`)
                - **Fire Hazard Factor:** `{1.0 if sev.fire_detected else 0.0}` (Weight: 0.20 $\\rightarrow$ Contribution: `{0.20 * (1.0 if sev.fire_detected else 0.0):.3f}`)
                - **Affected Zone Population Density:** `{sev.affected_zone_density}` (Weight: 0.20 $\\rightarrow$ Contribution: `{0.20 * sev.affected_zone_density:.3f}`)
                - **Composite Severity Score:** `{sev.score:.2f}` $\\implies$ **{sev.label} Severity**
                """)
