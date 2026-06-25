import streamlit as st

def apply_theme():
    st.markdown("""
    <style>
    /* ... keep your existing styles ... */

    /* ── Sidebar Navigation Overrides ── */
    [data-testid="stSidebar"] {
        background: #0b1957 !important;
        padding-top: 2rem;
    }
    
    /* Branding Area */
    .sidebar-logo {
        color: white;
        font-family: 'DM Serif Display', serif;
        font-size: 24px;
        margin-bottom: 5px;
        padding-left: 20px;
    }
    .sidebar-subtitle {
        color: #60a5fa;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 30px;
        padding-left: 20px;
    }

    /* Navigation Links Styling */
    .nav-link {
        display: flex;
        align-items: center;
        padding: 12px 20px;
        color: rgba(255,255,255,0.7) !important;
        text-decoration: none !important;
        font-size: 14px;
        transition: 0.2s;
    }
    .nav-link:hover {
        background: rgba(255,255,255,0.05);
        color: white !important;
    }
    .nav-link.active {
        background: rgba(255,255,255,0.1);
        color: white !important;
        border-right: 3px solid #60a5fa;
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    # Header/Branding
    st.sidebar.markdown("### BehaviourSense")
    st.sidebar.caption("AI INTERVIEW ANALYSIS")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    # Navigation items mapped to file paths
    # Ensure these match your actual filenames in the 'pages/' folder
    nav = {
        "Home": ("🏠", "app.py"),
        "Upload Interview": ("📤", "pages/2_Upload_Video"),
        "Live Interview": ("🎥", "pages/3_Live_Interview"),
        "Summary Report": ("📊", "pages/4_Summary_Report"),
        "Detailed Report": ("🔍", "pages/5_Detailed_Report")
    }

    for label, (icon, file) in nav.items():
        if st.sidebar.button(f"{icon} {label}"):
            # This handles the navigation to the corresponding .py file
            # If the file is in 'pages/', use f"pages/{file}"
            st.switch_page(f"{file}.py")


