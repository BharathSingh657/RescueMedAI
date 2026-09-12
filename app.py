"""
RescueMedAI - AI-Powered Disaster Response and Emergency Medical Decision Support System.
Main Streamlit Entrypoint.
"""
import streamlit as st

st.set_page_config(
    page_title="RescueMedAI — Disaster Decision Support System",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.frontend.main import main

if __name__ == "__main__":
    main()
