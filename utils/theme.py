import streamlit as st

def apply_theme():
    """Inject global CSS for BehaviourSense UI components."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@400;500;600;700&family=DM+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Page titles ── */
    .page-title {
        font-family: 'DM Serif Display', serif;
        font-size: 28px;
        color: #0b1957;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        line-height: 1.2;
    }
    .page-subtitle {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 24px;
        line-height: 1.5;
    }

    /* ── Cards ── */
    .bs-card {
        background: white;
        border-radius: 12px;
        border: 1px solid #dde6f0;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(11,25,87,0.06);
    }
    .bs-card-header {
        font-size: 13px;
        font-weight: 700;
        color: #0b1957;
        letter-spacing: 0.01em;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #f0f4f8;
    }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-teal {
        background: rgba(2,195,171,0.12);
        color: #0e9280;
        border: 1px solid rgba(2,195,171,0.3);
    }
    .badge-blue {
        background: rgba(66,107,194,0.12);
        color: #2e5db3;
        border: 1px solid rgba(66,107,194,0.3);
    }
    .badge-navy {
        background: rgba(11,25,87,0.08);
        color: #0b1957;
        border: 1px solid rgba(11,25,87,0.2);
    }

    /* ── Emotion bars ── */
    .emotion-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 9px;
    }
    .emotion-name {
        font-size: 12px;
        font-weight: 500;
        color: #334155;
        width: 70px;
        flex-shrink: 0;
    }
    .emotion-track {
        flex: 1;
        height: 8px;
        background: #f0f4f8;
        border-radius: 4px;
        overflow: hidden;
    }
    .emotion-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s ease;
    }
    .emotion-pct {
        font-size: 11px;
        color: #64748b;
        font-family: 'DM Mono', monospace;
        width: 38px;
        text-align: right;
        flex-shrink: 0;
    }

    /* ── Flags ── */
    .flag-item {
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
        border-left: 4px solid transparent;
    }
    .flag-critical {
        background: rgba(230,57,70,0.06);
        border-left-color: #e63946;
    }
    .flag-warning {
        background: rgba(244,162,97,0.08);
        border-left-color: #f4a261;
    }
    .flag-positive {
        background: rgba(34,197,94,0.07);
        border-left-color: #22c55e;
    }
    .flag-time {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: #94a3b8;
        background: #f0f4f8;
        padding: 2px 7px;
        border-radius: 4px;
    }
    .flag-title {
        font-size: 13px;
        font-weight: 600;
        color: #0b1957;
        margin: 5px 0 3px;
    }
    .flag-desc {
        font-size: 12px;
        color: #64748b;
        line-height: 1.5;
    }

    /* ── AI summary box ── */
    .ai-summary {
        background: linear-gradient(135deg, #0b1957 0%, #1a2f7a 100%);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        color: white;
    }
    .ai-summary div {
        color: rgba(255,255,255,0.85);
    }

    /* ── Divider ── */
    .bs-divider {
        border: none;
        border-top: 1px solid #eef2f8;
        margin: 24px 0;
    }

    /* ── Sidebar overrides ── */
    [data-testid="stSidebar"] {
        background: #0b1957 !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label {
        color: rgba(255,255,255,0.75) !important;
        font-size: 13px;
        font-weight: 500;
    }
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
