"""
Header and banner components for RescueMedAI.
"""
import streamlit as st


def render_hero_header():
    st.markdown("""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h1 class="hero-title">RescueMedAI — Disaster Decision Support Console</h1>
                <div class="hero-subtitle">
                    Integrated AI Prototype: Perception • Dynamic A* Path Planning • Priority Allocation • Bayesian Triage • SBAR Dispatch
                </div>
            </div>
            <div style="text-align: right;">
                <span style="background: #1e293b; border: 1px solid #38bdf8; color: #38bdf8; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">
                    OFFLINE PROTOTYPE
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_safety_notice(disclaimer: str):
    st.markdown(f"""
    <div class="safety-disclaimer-box">
        <span style="font-size: 1.2rem;">⚠️</span>
        <div><strong>MANDATORY SAFETY NOTICE:</strong> {disclaimer}</div>
    </div>
    """, unsafe_allow_html=True)


def render_simulation_notice(notice: str):
    st.markdown(f"""
    <div class="simulation-notice-box">
        ℹ️ <strong>PROTOTYPE SIMULATION NOTICE:</strong> {notice}
    </div>
    """, unsafe_allow_html=True)
