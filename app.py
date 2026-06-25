import streamlit as st
import runpy
from pathlib import Path

st.set_page_config(
    page_title="BehaviourSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",  # Forces it open on load
)

BASE = Path(__file__).resolve().parent

pages = {
    "🏠 Home": "app_pages/1_Homepage.py",
    "📤 Upload Interview": "app_pages/2_Upload_Video.py",
    "🎥 Live Interview": "app_pages/5_Live_Interview.py",
    "📊 Summary Report": "app_pages/3_Summary_Report.py",
    "🔍 Detailed Report": "app_pages/4_Detailed_Report.py",
    "🗂️ Past Interviews": "app_pages/6_Past_Interviews.py",
}

with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 16px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 8px;">
        <div style="font-family: 'DM Serif Display', serif; font-size: 20px; color: white; letter-spacing: -0.01em;">
            Behaviour<span style="color: #02c3ab;">Sense</span>
        </div>
        <div style="font-size: 10px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: rgba(255,255,255,0.35); margin-top: 2px;">
            AI Interview Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
    selected = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")

page_path = BASE / pages[selected]
runpy.run_path(str(page_path), run_name="__main__")



