"""
UI custom CSS injection for RescueMedAI Operations Command Center.
"""
import streamlit as st


def apply_custom_css():
    """Injects high-contrast, mission-critical tactical dark mode command center styling."""
    st.markdown("""
    <style>
    /* Dark Theme Command Center Canvas */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Hide Sidebar Completely for Full-Screen Tactical View */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Tactical Command Center Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 6px solid #0284c7;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 6px;
        line-height: 1.4;
    }

    .hero-status-tag {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .hero-alert-tag {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .command-bar-card {
        background-color: #131c2e;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }
    
    /* Safety Disclaimer Box */
    .safety-disclaimer-box {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid #dc2626;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 14px 0;
        color: #fca5a5;
        font-size: 0.86rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .simulation-notice-box {
        background: rgba(14, 165, 233, 0.08);
        border: 1px solid #0284c7;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 12px 0;
        color: #7dd3fc;
        font-size: 0.84rem;
    }
    
    /* Triage Badges */
    .triage-badge-red {
        background: linear-gradient(135deg, #b91c1c, #dc2626);
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: inline-block;
        box-shadow: 0 0 16px rgba(220, 38, 38, 0.6);
        border: 1px solid #ef4444;
    }
    
    .triage-badge-yellow {
        background: linear-gradient(135deg, #b45309, #d97706);
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: inline-block;
        box-shadow: 0 0 16px rgba(217, 119, 6, 0.5);
        border: 1px solid #f59e0b;
    }
    
    .triage-badge-green {
        background: linear-gradient(135deg, #047857, #059669);
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: inline-block;
        box-shadow: 0 0 16px rgba(5, 150, 105, 0.5);
        border: 1px solid #10b981;
    }

    /* Operations Metric Card */
    .ops-card {
        background-color: #131c2e;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .ops-card-title {
        color: #64748b;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .ops-card-value {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 700;
    }
    .ops-card-sub {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 4px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }
    
    /* Table headers & cells */
    .stDataFrame {
        border: 1px solid #1e293b !important;
        border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)
