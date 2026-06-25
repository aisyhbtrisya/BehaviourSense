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