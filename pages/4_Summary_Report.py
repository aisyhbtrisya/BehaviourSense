import streamlit as st
from utils.theme import apply_theme

st.set_page_config(
    page_title="BehaviourSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()