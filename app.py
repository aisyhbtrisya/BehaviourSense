"""
app.py  —  application SHELL / router (this is NOT a content page).

Think of it as the frame around everything:
  • it sets the page config,
  • it draws the sidebar (logo + the single navigation radio),
  • it routes to whichever screen you pick and runs that file.

Every actual SCREEN lives in pages/ — including the Home screen
(pages/1_Homepage.py). That's why you see "both": app.py is the frame, and
1_Homepage.py is the Home page's content. They are not duplicates.

We hide Streamlit's automatic pages/ sidebar nav (via CSS in utils/theme.py)
so there is only ONE navbar — this custom one.
"""

import runpy
from pathlib import Path

import streamlit as st

from utils.theme import apply_theme

st.set_page_config(
    page_title="BehaviourSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

BASE = Path(__file__).resolve().parent

# label -> file to run.  Edit the path on the right if your file names differ.
PAGES = {
    "🏠 Home": "pages/1_Homepage.py",
    "📤 Upload Interview": "pages/2_Upload_Video.py",
    "🎥 Live Interview": "pages/3_Live_Interview.py",
    "📊 Summary Report": "pages/4_Summary_Report.py",
    "🔍 Detailed Report": "pages/5_Detailed_Report.py"
}

with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 16px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 12px;">
        <div style="font-family: 'DM Serif Display', serif; font-size: 22px; color: white; letter-spacing: -0.01em;">
            Behaviour<span style="color: #02c3ab;">Sense</span>
        </div>
        <div style="font-size: 10px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: rgba(255,255,255,0.35); margin-top: 2px;">
            AI Interview Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # A stable `key` keeps the selection across reruns (e.g. when a child page
    # calls st.rerun()) and stops Streamlit auto-generating a colliding id.
    selected = st.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
        key="main_nav",
    )

page_path = BASE / PAGES[selected]

# Guard: never recurse into app.py, and fail gracefully if a file is missing.
if Path(PAGES[selected]).name == "app.py":
    st.error("Navigation target misconfigured (points at app.py).")
elif not page_path.exists():
    st.error(f"Page file not found: `{PAGES[selected]}`. "
             "Update its path in app.py's PAGES dict.")
else:
    runpy.run_path(str(page_path), run_name="__main__")



