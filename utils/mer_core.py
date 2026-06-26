"""
utils/mer_core.py
=================
Shared building blocks for Multimodal Emotion Recognition (FER + SER) used by
both the Upload page (2_Upload_Video.py) and the Live page (3_Live_Interview.py).

Keeping this in one place guarantees that the live session and the uploaded
video produce *identical* fusion + charts.

Contents:
  - Label maps (FER / SER) and the UNIFIED label space (their union)
  - to_unified_vector()         : project a model's probs into unified space
  - SelfAttention               : custom Keras layer required to load the FER model
  - load_fer_model / load_ser_resources / load_face_cascade  (cached)
  - extract_ser_features()      : SER feature pipeline (matches training, 14688-d)
  - fuse_results()              : weighted fusion on a shared timeline
  - build_probability_chart()   : interactive (zoom + hover) per-emotion chart
  - build_dominant_chart()      : interactive dominant-emotion confidence chart
  - overall_dominant_emotion()  : mean-probability winner
"""

import pickle

import cv2
import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# ============================================================================
# LABELS  (unified label space = union of FER + SER classes)
# ============================================================================
FER_LABELS = {0: "surprise", 1: "fear", 2: "disgust", 3: "happy", 4: "sad", 5: "angry", 6: "neutral"}
SER_LABELS = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad"}

UNIFIED_LABELS = sorted(set(FER_LABELS.values()) | set(SER_LABELS.values()))
N_UNIFIED = len(UNIFIED_LABELS)
UNIFIED_INDEX = {name: i for i, name in enumerate(UNIFIED_LABELS)}

# Consistent colours per emotion across all charts
EMOTION_COLORS = {
    "angry": "#e74c3c", "disgust": "#27ae60", "fear": "#8e44ad",
    "happy": "#f1c40f", "neutral": "#95a5a6", "sad": "#3498db", "surprise": "#e67e22",
}


def to_unified_vector(probs, label_map):
    """Project a model's raw probability vector into the unified label space."""
    out = np.zeros(N_UNIFIED)
    for idx, name in label_map.items():
        out[UNIFIED_INDEX[name]] = probs[idx]
    return out


# ============================================================================
# CUSTOM LAYER (required to load the FER model)
# ============================================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class SelfAttention(layers.Layer):
    def call(self, x):
        q = k = v = x
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        attention = tf.matmul(q, k, transpose_b=True)
        attention = attention / tf.math.sqrt(dk)
        attention = tf.nn.softmax(attention)
        return tf.matmul(attention, v)


# ============================================================================
# RESOURCE LOADING (cached)
# ============================================================================
@st.cache_resource
def load_fer_model():
    return load_model(
        "individual_models/models/patt_lite_rafdb.keras",
        custom_objects={"SelfAttention": SelfAttention},
        compile=False,
    )


@st.cache_resource
def load_ser_resources():
    model = load_model("individual_models/models/SER_CNN_BiLSTM_CREMAD.keras", compile=False)
    with open("individual_models/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("individual_models/models/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, scaler, encoder


@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def warm_up_models(fer_model, ser_model, scaler):
    """
    Run one dummy forward pass through each model on the CURRENT (main) thread.

    Why this matters for the live page: streamlit-webrtc calls
    VideoProcessor.recv()/AudioProcessor.recv() from its own background event
    loop thread, not Streamlit's main thread. The *first* call to a freshly
    loaded TensorFlow model triggers lazy op/kernel registration and graph
    tracing; doing that for the first time from a non-main thread inside a
    thin native wrapper (the webrtc event loop) is a known source of hard
    segfaults on minimal/cloud containers (no Python traceback — the whole
    process dies). Warming the models up here, on the main thread, before the
    stream starts means the callback thread's calls are never "the first
    ever" call, which avoids that class of crash entirely.
    """
    try:
        dummy_face = np.zeros((1, 224, 224, 3), dtype=np.float32)
        fer_model.predict(dummy_face, verbose=0)
    except Exception:
        pass  # best-effort warmup; real errors will surface on first real use

    try:
        dummy_audio = np.zeros(int(2.5 * 22050), dtype=np.float32)
        feats = extract_ser_features(dummy_audio, 22050)
        scaled = scaler.transform(feats.reshape(1, -1))
        ser_model.predict(scaled.reshape(1, -1, 1), verbose=0)
    except Exception:
        pass


# ============================================================================
# SER FEATURE EXTRACTION (matches training pipeline -> 14688-d vector)
# ============================================================================
def extract_ser_features(y, sr):
    ALPHA, BETA = 0.035, 0.7
    FRAME_LENGTH, HOP_LENGTH = 2048, 512

    def get_features(signal):
        zcr = np.squeeze(librosa.feature.zero_crossing_rate(y=signal, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH))
        rmse = np.squeeze(librosa.feature.rms(y=signal, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH))
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH).flatten()
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH).flatten()
        return np.concatenate((zcr, rmse, mfcc, chroma))

    f_oa = get_features(y)
    f_na = get_features(y + ALPHA * np.random.normal(0, 1, len(y)))
    # librosa >= 0.10 requires y= as keyword
    f_pa = get_features(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=BETA))
    f_comb = get_features(librosa.effects.pitch_shift(y=y + ALPHA * np.random.normal(0, 1, len(y)), sr=sr, n_steps=BETA))

    return np.nan_to_num(np.concatenate((f_oa, f_na, f_pa, f_comb)), nan=0.0)


# ============================================================================
# FUSION — resample both streams onto a shared timeline, then weight & sum
# ============================================================================
def fuse_results(fer_result, ser_result, w_fer, w_ser, n_points=200):
    """
    Interpolates both probability streams onto a common timeline so they can
    be combined even though FER and SER are sampled at different rates.
    Missing modality (e.g. no audio track) contributes a zero vector.
    """
    fer_ts, fer_probs = fer_result["timestamps"], fer_result["probs"]
    ser_ts, ser_probs = ser_result["timestamps"], ser_result["probs"]

    all_ts = np.concatenate([t for t in (fer_ts, ser_ts) if len(t) > 0]) if (len(fer_ts) or len(ser_ts)) else np.array([0.0])
    if len(all_ts) == 0:
        all_ts = np.array([0.0])
    t_min, t_max = float(np.min(all_ts)), float(np.max(all_ts))
    if t_max <= t_min:
        t_max = t_min + 1.0
    common_ts = np.linspace(t_min, t_max, n_points)

    def resample(ts, probs):
        if len(ts) == 0:
            return np.zeros((len(common_ts), N_UNIFIED))
        if len(ts) == 1:
            return np.tile(probs[0], (len(common_ts), 1))
        out = np.zeros((len(common_ts), N_UNIFIED))
        for c in range(N_UNIFIED):
            out[:, c] = np.interp(common_ts, ts, probs[:, c])
        return out

    fer_resampled = resample(fer_ts, fer_probs)
    ser_resampled = resample(ser_ts, ser_probs)

    fused = w_fer * fer_resampled + w_ser * ser_resampled
    # Renormalise so each timestep sums to 1 (keeps a clean probability distribution).
    # NOTE: renormalising never changes the argmax, so the predicted emotion is
    # the same as the raw weighted sum in the user's spec; it only rescales the
    # y-axis for display.
    row_sums = fused.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    fused_normalized = fused / row_sums

    return {
        "timestamps": common_ts,
        "fer_resampled": fer_resampled,
        "ser_resampled": ser_resampled,
        "fused": fused_normalized,
    }


# ============================================================================
# INTERACTIVE PLOTLY CHARTS (zoom + hover)
# ============================================================================
def build_probability_chart(timestamps, probs_matrix, title):
    fig = go.Figure()
    for name in UNIFIED_LABELS:
        idx = UNIFIED_INDEX[name]
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=probs_matrix[:, idx],
            mode="lines",
            name=name.capitalize(),
            line=dict(color=EMOTION_COLORS.get(name, "#333333"), width=2),
            hovertemplate=f"<b>{name.capitalize()}</b><br>Time: %{{x:.2f}}s<br>Confidence: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        hovermode="x unified",
        legend_title="Emotion",
        height=480,
        dragmode="zoom",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def build_dominant_chart(timestamps, probs_matrix, title):
    dominant_idx = np.argmax(probs_matrix, axis=1)
    dominant_conf = np.max(probs_matrix, axis=1)
    dominant_names = [UNIFIED_LABELS[i] for i in dominant_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=dominant_conf,
        mode="lines+markers",
        line=dict(color="#8e44ad", width=2),
        marker=dict(size=6),
        text=dominant_names,
        hovertemplate="<b>%{text}</b><br>Time: %{x:.2f}s<br>Confidence: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Confidence of Dominant Emotion",
        yaxis_range=[0, 1],
        hovermode="closest",
        height=420,
        dragmode="zoom",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def overall_dominant_emotion(probs_matrix):
    if probs_matrix.shape[0] == 0:
        return "N/A", 0.0
    mean_probs = np.mean(probs_matrix, axis=0)
    idx = int(np.argmax(mean_probs))
    return UNIFIED_LABELS[idx], float(mean_probs[idx])


# ============================================================================
# CROSS-PAGE RESULT SHARING + REPORT RENDERERS
# ----------------------------------------------------------------------------
# The Upload and Live pages run the analysis and STORE the results in
# st.session_state. The Detailed Report and Summary Report pages READ them back
# and render the visuals. This keeps the heavy graphs off the capture pages and
# in the report pages, exactly as requested.
# ============================================================================
FUSION_WEIGHTS = (0.6, 0.4)  # (w_fer, w_ser) — fixed 60/40 fusion, app-wide


def store_results(fer_result, ser_result, fusion, source):
    """Persist the latest analysis so the report pages can display it."""
    st.session_state["mer_fer_result"] = fer_result
    st.session_state["mer_ser_result"] = ser_result
    st.session_state["mer_fusion"] = fusion
    st.session_state["mer_source"] = source
    st.session_state["mer_has_result"] = True


def get_results():
    """Return the latest stored analysis, or None if nothing has run yet."""
    if not st.session_state.get("mer_has_result"):
        return None
    return {
        "fer_result": st.session_state.get("mer_fer_result"),
        "ser_result": st.session_state.get("mer_ser_result"),
        "fusion": st.session_state.get("mer_fusion"),
        "source": st.session_state.get("mer_source", "—"),
    }


def build_distribution_donut(probs_matrix, title):
    """Donut chart of the average emotion distribution over the whole session."""
    if probs_matrix is None or probs_matrix.shape[0] == 0:
        mean = np.zeros(N_UNIFIED)
    else:
        mean = probs_matrix.mean(axis=0)

    total = mean.sum()
    if total > 0:
        mean = mean / total  # clean probability distribution that sums to 1

    labels, values, colors = [], [], []
    for name in UNIFIED_LABELS:
        v = float(mean[UNIFIED_INDEX[name]])
        if v > 0.005:  # hide negligible slices
            labels.append(name.capitalize())
            values.append(v)
            colors.append(EMOTION_COLORS.get(name, "#333333"))

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="percent",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
        sort=True,
        direction="clockwise",
    ))
    fig.update_layout(
        title=title,
        height=440,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="v", yanchor="middle", y=0.5),
    )
    return fig


def _no_results_notice():
    st.info(
        "No analysis yet. Run an interview on the **Upload Interview** or "
        "**Live Interview** page first, then come back here."
    )


def render_detailed_report():
    """Interactive time-series breakdown (Multimodal / FER / SER)."""
    results = get_results()
    if results is None:
        _no_results_notice()
        return

    fer_result = results["fer_result"]
    ser_result = results["ser_result"]
    fusion = results["fusion"]
    st.caption(f"Showing the most recent result · source: **{results['source']}**")

    mode = st.radio("Select view", ["Multimodal", "FER", "SER"],
                    horizontal=True, key="detailed_mode")

    if mode == "Multimodal":
        dom_label, dom_conf = overall_dominant_emotion(fusion["fused"])
        st.metric("Final Predicted Emotion (Fused)", dom_label.capitalize(), f"{dom_conf:.1%} confidence")
        st.plotly_chart(build_probability_chart(fusion["timestamps"], fusion["fused"],
                        "Multimodal Fused Emotion Probabilities Over Time"), use_container_width=True)
        st.plotly_chart(build_dominant_chart(fusion["timestamps"], fusion["fused"],
                        "Multimodal Dominant Emotion Confidence Over Time"), use_container_width=True)

    elif mode == "FER":
        if fer_result["probs"].shape[0] == 0:
            st.info("No FER data available.")
        else:
            dom_label, dom_conf = overall_dominant_emotion(fer_result["probs"])
            st.metric("Final Predicted Emotion (FER only)", dom_label.capitalize(), f"{dom_conf:.1%} confidence")
            st.plotly_chart(build_probability_chart(fer_result["timestamps"], fer_result["probs"],
                            "Facial Emotion Probabilities Over Time"), use_container_width=True)
            st.plotly_chart(build_dominant_chart(fer_result["timestamps"], fer_result["probs"],
                            "FER Dominant Emotion Confidence Over Time"), use_container_width=True)
            if "face_found" in fer_result and len(fer_result["face_found"]):
                n_no_face = int((~fer_result["face_found"]).sum())
                if n_no_face:
                    st.caption(f"No face detected in {n_no_face} of {len(fer_result['face_found'])} sampled frames (shown as zero confidence).")

    else:  # SER
        if ser_result["probs"].shape[0] == 0:
            st.info("No SER data available (no or insufficient audio).")
        else:
            dom_label, dom_conf = overall_dominant_emotion(ser_result["probs"])
            st.metric("Final Predicted Emotion (SER only)", dom_label.capitalize(), f"{dom_conf:.1%} confidence")
            st.plotly_chart(build_probability_chart(ser_result["timestamps"], ser_result["probs"],
                            "Speech Emotion Probabilities Over Time"), use_container_width=True)
            st.plotly_chart(build_dominant_chart(ser_result["timestamps"], ser_result["probs"],
                            "SER Dominant Emotion Confidence Over Time"), use_container_width=True)

    st.caption("Tip: drag to zoom, hover for exact confidence, use the range slider, double-click to reset.")


def render_summary_report():
    """High-level summary: dominant emotion per modality + a distribution donut."""
    results = get_results()
    if results is None:
        _no_results_notice()
        return

    fer_result = results["fer_result"]
    ser_result = results["ser_result"]
    fusion = results["fusion"]
    st.caption(f"Showing the most recent result · source: **{results['source']}**")

    fus_lbl, fus_conf = overall_dominant_emotion(fusion["fused"])
    fer_lbl, fer_conf = overall_dominant_emotion(fer_result["probs"])
    ser_lbl, ser_conf = overall_dominant_emotion(ser_result["probs"])

    st.subheader("Dominant emotion")
    m1, m2, m3 = st.columns(3)
    m1.metric("Multimodal (Fused)", fus_lbl.capitalize(), f"{fus_conf:.1%}")
    m2.metric("Facial (FER)", fer_lbl.capitalize(), f"{fer_conf:.1%}")
    m3.metric("Speech (SER)", ser_lbl.capitalize(), f"{ser_conf:.1%}")

    st.subheader("Emotion distribution")
    which = st.radio("Distribution for", ["Multimodal", "FER", "SER"],
                     horizontal=True, key="summary_donut_mode")
    if which == "Multimodal":
        probs = fusion["fused"]
    elif which == "FER":
        probs = fer_result["probs"]
    else:
        probs = ser_result["probs"]

    if probs is None or probs.shape[0] == 0:
        st.info(f"No {which} data to chart.")
    else:
        st.plotly_chart(build_distribution_donut(probs, f"{which} — Emotion Distribution"),
                        use_container_width=True)