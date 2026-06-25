
"""
2_Upload_Video.py
==================
Multimodal Emotion Recognition (FER + SER) with Weighted Fusion.

Pipeline:
1. User uploads ONE video file.
2. Facial Emotion Recognition (FER) runs on sampled frames (PAtt-Lite / RAF-DB, 7 classes).
3. Speech Emotion Recognition (SER) runs on the extracted audio track
   (CNN-BiLSTM / CREMA-D, 6 classes) using a sliding window.
4. Both probability streams are resampled onto a shared timeline and projected
   into a UNIFIED label space (union of both label sets; a model contributes
   0.0 probability for any class it doesn't predict).
5. A weighted fusion combines them: fused = w_fer * FER_probs + w_ser * SER_probs
6. User clicks "Display Result" and chooses Multimodal / FER / SER via radio
   buttons to inspect an interactive (zoom + hover) Plotly chart of emotion
   probabilities over time, plus the final dominant emotion per mode.

Notes on fixes vs. the original two scripts this was merged from:
- Original SER script wrote every upload to a ".mp4" temp file even for
  audio-only uploads (.wav/.mp3); this preserves the real suffix instead.
- Original SER script called librosa.effects.pitch_shift(y, sr=sr, n_steps=BETA)
  positionally for `y`, which raises a TypeError on modern librosa
  (>=0.10) since `y` must be passed as a keyword. Fixed here.
- FER and SER use different label sets; fusion is only meaningful in a
  shared label space, so we map both into the union of labels.
"""

import os
import tempfile

import cv2
import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from utils.theme import apply_theme
from moviepy.editor import AudioFileClip, VideoFileClip
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

apply_theme()

st.set_page_config(page_title="Multimodal Emotion Recognition", layout="wide")

# ============================================================================
# 1. LABELS  (unified label space = union of FER + SER classes)
# ============================================================================
FER_LABELS = {0: "surprise", 1: "fear", 2: "disgust", 3: "happy", 4: "sad", 5: "angry", 6: "neutral"}
SER_LABELS = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad"}

UNIFIED_LABELS = sorted(set(FER_LABELS.values()) | set(SER_LABELS.values()))
N_UNIFIED = len(UNIFIED_LABELS)
UNIFIED_INDEX = {name: i for i, name in enumerate(UNIFIED_LABELS)}

# Consistent colors per emotion across all three charts
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
# 2. CUSTOM LAYER (required to load the FER model)
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
# 3. RESOURCE LOADING (cached)
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
    import pickle

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
# 4. SER FEATURE EXTRACTION (matches training pipeline)
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
    # FIX: librosa >=0.10 requires y= as keyword
    f_pa = get_features(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=BETA))
    f_comb = get_features(librosa.effects.pitch_shift(y=y + ALPHA * np.random.normal(0, 1, len(y)), sr=sr, n_steps=BETA))

    return np.nan_to_num(np.concatenate((f_oa, f_na, f_pa, f_comb)), nan=0.0)


# ============================================================================
# 5. FER PIPELINE — runs over sampled video frames
# ============================================================================
def run_fer_pipeline(video_path, sample_interval, fer_model, face_cascade, progress_cb=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip_interval = max(1, int(fps * sample_interval))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timestamps, unified_probs_list, face_found = [], [], []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip_interval == 0:
            timestamp = round(frame_count / fps, 2)
            timestamps.append(timestamp)

            if progress_cb is not None and total_frames > 0:
                progress_cb(min(frame_count / total_frames, 1.0), f"FER: analyzing frame at {timestamp}s")

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = image_rgb[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (224, 224))
                face_normalized = face_resized / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)

                prediction = fer_model.predict(face_input, verbose=0)[0]
                unified_probs_list.append(to_unified_vector(prediction, FER_LABELS))
                face_found.append(True)
            else:
                # No face detected -> zero-confidence vector (per user spec)
                unified_probs_list.append(np.zeros(N_UNIFIED))
                face_found.append(False)

        frame_count += 1

    cap.release()
    return {
        "timestamps": np.array(timestamps, dtype=float),
        "probs": np.array(unified_probs_list) if unified_probs_list else np.zeros((0, N_UNIFIED)),
        "face_found": np.array(face_found, dtype=bool),
    }


# ============================================================================
# 6. SER PIPELINE — sliding window over the extracted audio track
# ============================================================================

def run_ser_pipeline(video_path, ser_model, scaler, progress_cb=None):
    temp_wav = tempfile.mktemp(suffix=".wav")
    
    try:
        # Load the video file
        clip = VideoFileClip(video_path)
        
        # Check if the video actually has an audio track
        if clip.audio is None:
            clip.close()
            return {"timestamps": np.array([]), "probs": np.zeros((0, N_UNIFIED)), "has_audio": False}
        
        # Extract audio
        clip.audio.write_audiofile(temp_wav, fps=22050, logger=None, verbose=False)
        clip.close()
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return {"timestamps": np.array([]), "probs": np.zeros((0, N_UNIFIED)), "has_audio": False}

    # Load audio for processing
    audio, sr = librosa.load(temp_wav, sr=22050)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    # ... (Rest of your existing sliding window logic remains exactly the same) ...
    window_samples = int(2.5 * sr)
    step_samples = int(1.0 * sr)
    # ... etc


# ============================================================================
# 7. FUSION — resample both streams onto a shared timeline, then weight & sum
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
    # Renormalize so each timestep sums to 1 (keeps it a clean probability distribution)
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
# 8. INTERACTIVE PLOTLY CHARTS (zoom + hover)
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
# 9. STREAMLIT UI
# ============================================================================
st.title("🎭 Multimodal Emotion Recognition (FER + SER)")
st.caption("Upload a video. Facial and speech emotion models run independently, then are combined via weighted fusion.")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Preserve the real file extension (fixes a bug in the original SER app
    # where every upload was forced to .mp4 even for audio-only files).
    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_video.read())
        video_path = tfile.name

    st.video(video_path)

    st.subheader("⚙️ Settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        sample_interval = st.slider("FER sampling interval (seconds)", 0.1, 3.0, 0.5, 0.1)
    with c2:
        w_fer = st.slider("FER weight", 0.0, 1.0, 0.6, 0.05)
    with c3:
        w_ser = round(1.0 - w_fer, 2)
        st.metric("SER weight (auto)", w_ser)

    if st.button("🚀 Run Analysis", type="primary"):
        fer_model = load_fer_model()
        ser_model, scaler, _encoder = load_ser_resources()
        face_cascade = load_face_cascade()

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def make_cb(lo, hi):
            def cb(frac, msg):
                progress_bar.progress(lo + frac * (hi - lo))
                status_text.text(msg)
            return cb

        with st.spinner("Running facial emotion recognition..."):
            fer_result = run_fer_pipeline(video_path, sample_interval, fer_model, face_cascade, make_cb(0.0, 0.5))

        with st.spinner("Running speech emotion recognition..."):
            ser_result = run_ser_pipeline(video_path, ser_model, scaler, make_cb(0.5, 1.0))

        # Robust check
        if ser_result is None or not ser_result.get("has_audio", True):
            st.warning("No audio track detected or analysis failed. SER results unavailable.")
            # Set a dummy empty result so the rest of the code doesn't break
            ser_result = {"timestamps": np.array([]), "probs": np.zeros((0, N_UNIFIED)), "has_audio": False}

        progress_bar.progress(1.0)
        status_text.text("Analysis complete.")
        progress_bar.empty()
        status_text.empty()

        if not ser_result.get("has_audio", True):
            st.warning("No audio track detected in this video — SER results will be empty (zero confidence).")
        if fer_result["probs"].shape[0] > 0 and not fer_result["face_found"].any():
            st.warning("No face was detected in any sampled frame — FER results will be empty (zero confidence).")

        fusion = fuse_results(fer_result, ser_result, w_fer, w_ser)

        # Persist everything needed for the result section across reruns
        st.session_state["fer_result"] = fer_result
        st.session_state["ser_result"] = ser_result
        st.session_state["fusion"] = fusion
        st.session_state["analysis_done"] = True

    # ------------------------------------------------------------------
    # RESULTS SECTION
    # ------------------------------------------------------------------
    if st.session_state.get("analysis_done"):
        st.divider()
        st.subheader("📊 Results")

        if st.button("Display Result"):
            st.session_state["show_results"] = True

        if st.session_state.get("show_results"):
            mode = st.radio(
                "Select view",
                options=["Multimodal", "FER", "SER"],
                horizontal=True,
                key="result_mode",
            )

            fer_result = st.session_state["fer_result"]
            ser_result = st.session_state["ser_result"]
            fusion = st.session_state["fusion"]

            if mode == "Multimodal":
                dom_label, dom_conf = overall_dominant_emotion(fusion["fused"])
                st.metric("Final Predicted Emotion (Fused)", dom_label.capitalize(), f"{dom_conf:.1%} confidence")
                st.plotly_chart(
                    build_probability_chart(fusion["timestamps"], fusion["fused"], "Multimodal Fused Emotion Probabilities Over Time"),
                    use_container_width=True,
                )
                st.plotly_chart(
                    build_dominant_chart(fusion["timestamps"], fusion["fused"], "Multimodal Dominant Emotion Confidence Over Time"),
                    use_container_width=True,
                )

            elif mode == "FER":
                if fer_result["probs"].shape[0] == 0:
                    st.info("No FER data available for this video.")
                else:
                    dom_label, dom_conf = overall_dominant_emotion(fer_result["probs"])
                    st.metric("Final Predicted Emotion (FER only)", dom_label.capitalize(), f"{dom_conf:.1%} confidence")
                    st.plotly_chart(
                        build_probability_chart(fer_result["timestamps"], fer_result["probs"], "Facial Emotion Probabilities Over Time"),
                        use_container_width=True,
                    )
                    st.plotly_chart(
                        build_dominant_chart(fer_result["timestamps"], fer_result["probs"], "FER Dominant Emotion Confidence Over Time"),
                        use_container_width=True,
                    )
                    n_no_face = int((~fer_result["face_found"]).sum())
                    if n_no_face:
                        st.caption(f"No face detected in {n_no_face} of {len(fer_result['face_found'])} sampled frames (shown as zero confidence).")

            elif mode == "SER":
                if ser_result["probs"].shape[0] == 0:
                    st.info("No SER data available for this video (likely no audio track).")
                else:
                    dom_label, dom_conf = overall_dominant_emotion(ser_result["probs"])
                    st.metric("Final Predicted Emotion (SER only)", dom_label.capitalize(), f"{dom_conf:.1%} confidence")
                    st.plotly_chart(
                        build_probability_chart(ser_result["timestamps"], ser_result["probs"], "Speech Emotion Probabilities Over Time"),
                        use_container_width=True,
                    )
                    st.plotly_chart(
                        build_dominant_chart(ser_result["timestamps"], ser_result["probs"], "SER Dominant Emotion Confidence Over Time"),
                        use_container_width=True,
                    )

            st.caption("Tip: drag on the chart to zoom in on a time range, hover over any line to see exact confidence, and use the range slider below the x-axis. Double-click the chart to reset zoom.")
else:
    st.session_state["analysis_done"] = False
    st.session_state["show_results"] = False