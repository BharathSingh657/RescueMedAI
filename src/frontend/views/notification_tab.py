"""
Hospital Pre-Arrival Notification View (Tab 5).
Displays standardized SBAR pre-arrival dispatch tickets, hospital preparedness directives,
and exports professional PDF emergency reports.
"""
import streamlit as st
from src.backend.state import SIMULATED_DATA_NOTICE
from src.backend.hospital.pdf_generator import HospitalPDFReportGenerator
from src.frontend.components.header import render_simulation_notice


def render_notification_tab():
    state = st.session_state.state

    st.subheader("🏥 Receiving Emergency Department Pre-Arrival Console")
    st.caption("Standardized SBAR / MIST pre-arrival dispatch tickets, facility preparedness directives, and professional PDF report generation.")
    
    render_simulation_notice(SIMULATED_DATA_NOTICE)

    if state.hospital_notification:
        notif = state.hospital_notification
        hcol1, hcol2 = st.columns([1.3, 1.0])

        with hcol1:
            st.markdown("##### 📋 Standardized SBAR Pre-Arrival Dispatch Ticket")
            st.markdown(f"""
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 18px;">
                {notif.sbar_formatted_text}
            </div>
            """, unsafe_allow_html=True)

        with hcol2:
            st.markdown("##### 🚨 Facility Preparedness Directives")
            for item in notif.recommended_preparedness:
                st.markdown(f"☑️ **{item}**")

            st.markdown("---")
            st.markdown("##### 📤 Professional Emergency Handoff Export")
            
            # PDF Report Export
            try:
                pdf_bytes = HospitalPDFReportGenerator.generate(state, notif)
                st.download_button(
                    label="📄 Download Professional Handoff Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"handoff_{notif.case_id}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF report: {e}")

            # Markdown Report Export
            st.download_button(
                label="📥 Download Handoff Report (Markdown)",
                data=notif.sbar_formatted_text,
                file_name=f"handoff_{notif.case_id}.md",
                mime="text/markdown",
                use_container_width=True
            )

            st.markdown("---")
            st.markdown("##### 📡 Emergency Radio Broadcast Telemetry")
            if st.button("📢 Simulate Emergency Radio Broadcast Handoff", use_container_width=True):
                st.toast(f"Handoff broadcast for {notif.case_id} transmitted to {notif.hospital_name}!", icon="🚑")
