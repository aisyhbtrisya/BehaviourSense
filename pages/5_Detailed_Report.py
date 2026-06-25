"""
5_Detailed_Report.py
====================
Full interactive time-series breakdown of the most recent interview
(Multimodal / FER / SER). The analysis itself is run on the Upload or Live
page; this page just reads the stored result and renders the graphs.
"""

import streamlit as st
from utils.theme import apply_theme
from utils.mer_core import render_detailed_report

# set_page_config can only run once; guard so it works standalone AND when
# launched through app.py (which already calls it before runpy).
try:
    st.set_page_config(page_title="BehaviourSense AI", page_icon="🧠",
                       layout="wide", initial_sidebar_state="expanded")
except Exception:
    pass

apply_theme()

st.title("🔍 Detailed Report")
st.caption("Interactive emotion timelines for the most recent interview.")

render_detailed_report()