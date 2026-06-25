"""
4_Summary_Report.py
===================
High-level summary of the most recent interview: the dominant emotion for
each modality (Multimodal / FER / SER) plus a donut chart of the overall
emotion distribution. Reads the result stored by the Upload or Live page.
"""

import streamlit as st
from utils.theme import apply_theme
from utils.mer_core import render_summary_report

try:
    st.set_page_config(page_title="BehaviourSense AI", page_icon="🧠",
                       layout="wide", initial_sidebar_state="expanded")
except Exception:
    pass

apply_theme()

st.title("📊 Summary Report")
st.caption("At-a-glance emotion summary for the most recent interview.")

render_summary_report()