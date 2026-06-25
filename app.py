import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.theme import apply_theme

apply_theme()
render_sidebar():

# ── Hero ────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #0b1957 0%, #426bc2 100%);
            border-radius: 16px; padding: 48px 40px; margin-bottom: 28px; position: relative; overflow: hidden;">
    <div style="position: absolute; top: -60px; right: -60px; width: 300px; height: 300px;
                background: rgba(2,195,171,0.08); border-radius: 50%;"></div>
    <div style="position: absolute; bottom: -40px; right: 100px; width: 200px; height: 200px;
                background: rgba(255,255,255,0.04); border-radius: 50%;"></div>
    <div style="position: relative; z-index: 1;">
        <div style="display: inline-flex; align-items: center; gap: 6px; background: rgba(2,195,171,0.15);
                    border: 1px solid rgba(2,195,171,0.3); border-radius: 20px;
                    padding: 4px 12px; margin-bottom: 16px;">
            <span style="color: #02c3ab; font-size: 10px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;">
                🧠 Multimodal AI Analysis
            </span>
        </div>
        <h1 style="font-family: 'DM Serif Display', serif; font-size: 42px; color: white;
                   line-height: 1.15; margin: 0 0 14px; letter-spacing: -0.02em;">
            Understand what words<br><em style="color: #02c3ab;">don't say</em>
        </h1>
        <p style="font-size: 15px; color: rgba(255,255,255,0.65); max-width: 520px;
                  line-height: 1.65; margin: 0 0 28px;">
            BehaviourSense analyses facial expressions, speech tone, and microexpressions
            from interview recordings to surface emotional patterns invisible to the naked eye.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── How it works ────────────────────────────────────────────────
st.markdown("""
<div style="font-size: 11px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
            color: #64748b; margin-bottom: 16px;">HOW IT WORKS</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="bs-card" style="border-top: 3px solid #426bc2;">
        <div style="font-size: 28px; margin-bottom: 12px;">👁️</div>
        <div style="font-weight: 700; color: #0b1957; font-size: 15px; margin-bottom: 8px;">
            Facial Emotion Recognition
        </div>
        <div style="font-size: 13px; color: #64748b; line-height: 1.6;">
            MobileNet + Patch Attention analyses every frame to detect
            sustained facial expressions across 7 emotion categories.
        </div>
        <div style="margin-top: 14px;">
            <span class="badge badge-blue">MobileNet V1</span>
            <span class="badge badge-blue" style="margin-left: 6px;">RAF-DB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="bs-card" style="border-top: 3px solid #02c3ab;">
        <div style="font-size: 28px; margin-bottom: 12px;">🎙️</div>
        <div style="font-weight: 700; color: #0b1957; font-size: 15px; margin-bottom: 8px;">
            Speech Emotion Recognition
        </div>
        <div style="font-size: 13px; color: #64748b; line-height: 1.6;">
            CNN + Attention BiLSTM extracts MFCC, chroma, and mel features
            from voice to detect emotional tone and stress patterns.
        </div>
        <div style="margin-top: 14px;">
            <span class="badge badge-teal">CNN BiLSTM</span>
            <span class="badge badge-teal" style="margin-left: 6px;">RAVDESS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="bs-card" style="border-top: 3px solid #8b5cf6;">
        <div style="font-size: 28px; margin-bottom: 12px;">⚡</div>
        <div style="font-weight: 700; color: #0b1957; font-size: 15px; margin-bottom: 8px;">
            Microexpression Recognition
        </div>
        <div style="font-size: 13px; color: #64748b; line-height: 1.6;">
            Optical Flow + CNN detects suppressed emotional leakage —
            involuntary microexpressions lasting under 200 milliseconds.
        </div>
        <div style="margin-top: 14px;">
            <span class="badge" style="background: rgba(139,92,246,0.12); color: #6d28d9; border: 1px solid rgba(139,92,246,0.3);">Optical Flow</span>
            <span class="badge" style="background: rgba(139,92,246,0.12); color: #6d28d9; border: 1px solid rgba(139,92,246,0.3); margin-left: 6px;">CASME II</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='bs-divider'></div>", unsafe_allow_html=True)

# ── Model performance ───────────────────────────────────────────
st.markdown("""
<div style="font-size: 11px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
            color: #64748b; margin-bottom: 16px;">MODEL PERFORMANCE</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("FER Accuracy",  "75.65%", "RAF-DB dataset")
with c2:
    st.metric("SER Accuracy",  "62.00%", "RAVDESS dataset")
with c3:
    st.metric("MER Accuracy",  "—",      "Training in progress")
with c4:
    st.metric("Fusion Weight", "45/45/10", "FER / SER / MER")

st.markdown("<div class='bs-divider'></div>", unsafe_allow_html=True)

# ── Disclaimer ──────────────────────────────────────────────────
st.markdown("""
<div style="background: #e9f3ff; border: 1px solid rgba(66,107,194,0.3); border-radius: 10px;
            padding: 14px 18px; display: flex; gap: 12px; align-items: flex-start;">
    <span style="font-size: 18px;">ℹ️</span>
    <div>
        <div style="font-size: 13px; font-weight: 700; color: #0b1957;">Decision Support Tool Only</div>
        <div style="font-size: 12px; color: #334155; margin-top: 3px; line-height: 1.6;">
            BehaviourSense is designed to assist human decision-making, not replace it.
            All outputs should be interpreted by a qualified investigator or journalist.
            Emotion detection cannot determine deception or intent directly.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)