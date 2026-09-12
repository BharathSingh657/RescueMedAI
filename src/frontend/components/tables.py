"""
Table and Metric rendering utilities for RescueMedAI.
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any


def render_metric_card(title: str, value: str, sub: str = ""):
    st.markdown(f"""
    <div class="ops-card">
        <div class="ops-card-title">{title}</div>
        <div class="ops-card-value">{value}</div>
        <div class="ops-card-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def safe_render_image(img, caption=None):
    """Safely renders image across all Streamlit versions."""
    try:
        st.image(img, caption=caption, use_column_width=True)
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)
