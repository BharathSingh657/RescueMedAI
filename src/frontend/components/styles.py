"""
UI custom CSS injection for RescueMedAI Operations Dashboard.
"""
import streamlit as st


def apply_custom_css():
    """Injects high-contrast, mission-critical dark mode styling."""
    st.markdown("""
    <style>
    /* Dark Theme Mission Control Styling */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Top Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 6px solid #0284c7;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        line-height: 1.4;
    }
    
    /* Safety Disclaimer Box */
    .safety-disclaimer-box {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid #dc2626;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 16px 0;
        color: #fca5a5;
        font-size: 0.88rem;
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
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .ops-card-value {
        color: #f8fafc;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .ops-card-sub {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)
