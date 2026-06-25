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
import streamlit as st
from moviepy import VideoFileClip

from utils.theme import apply_theme
from utils.mer_core import (
    FER_LABELS,
    SER_LABELS,
    N_UNIFIED,
    FUSION_WEIGHTS,
    to_unified_vector,
    load_fer_model,
    load_ser_resources,
    load_face_cascade,
    extract_ser_features,
    fuse_results,
    store_results,
    overall_dominant_emotion,
)

# st.set_page_config can only be called once per session AND must be the first
# Streamlit command. When this page is launched through app.py (which already
# calls it before runpy), a second call raises StreamlitAPIException. Guarding
# it lets the page work both standalone and when embedded by app.py.
try:
    st.set_page_config(
        page_title="BehaviourSense AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

apply_theme()


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
    """
    Extract the audio track from the uploaded video and run Speech Emotion
    Recognition over it with a sliding window. Returns probabilities projected
    into the UNIFIED label space so they can be fused with FER.

    Returns a dict:
        {
            "timestamps": np.ndarray[float],          # window centre times (s)
            "probs":      np.ndarray[n_windows, N_UNIFIED],
            "has_audio":  bool,
        }
    """
    empty = {"timestamps": np.array([]), "probs": np.zeros((0, N_UNIFIED)), "has_audio": False}
    temp_wav = tempfile.mktemp(suffix=".wav")

    # ---- 1. Extract the audio track ------------------------------------
    try:
        clip = VideoFileClip(video_path)

        # No audio stream at all -> nothing for SER to do.
        if clip.audio is None:
            clip.close()
            return empty

        # NOTE: moviepy >= 2.0 removed the `verbose` argument from
        # write_audiofile(); passing it raises TypeError. Use `logger=None`
        # to stay quiet instead.
        clip.audio.write_audiofile(temp_wav, fps=22050, logger=None)
        clip.close()
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return empty

    # ---- 2. Load the waveform -----------------------------------------
    try:
        audio, sr = librosa.load(temp_wav, sr=22050)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return empty
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

    if audio is None or len(audio) == 0:
        return empty

    # ---- 3. Sliding window (must match the SER training config) --------
    # 2.5 s windows, stepped every 1.0 s, sr = 22050 -> feature vector of
    # length 14688, which is exactly what scaler / model expect.
    window_samples = int(2.5 * sr)
    step_samples = int(1.0 * sr)

    # If the clip is shorter than one window, pad it so we still get a
    # single prediction (keeps short uploads from returning empty SER).
    if len(audio) < window_samples:
        audio = np.pad(audio, (0, window_samples - len(audio)), mode="constant")

    starts = list(range(0, len(audio) - window_samples + 1, step_samples))
    if not starts:
        starts = [0]

    timestamps, unified_probs_list = [], []
    n_windows = len(starts)

    for i, start in enumerate(starts):
        window = audio[start:start + window_samples]

        # Feature extraction -> scale -> predict (identical to the
        # standalone SER app that already works).
        features = extract_ser_features(window, sr)
        scaled = scaler.transform(features.reshape(1, -1))
        preds = ser_model.predict(scaled.reshape(1, -1, 1), verbose=0)[0]

        # Project the 6-class SER distribution into the unified label space.
        unified_probs_list.append(to_unified_vector(preds, SER_LABELS))

        # Window-centre timestamp (seconds), matching the standalone app.
        timestamps.append((start + window_samples / 2) / sr)

        if progress_cb is not None:
            progress_cb((i + 1) / n_windows, f"SER: analyzing window at {timestamps[-1]:.1f}s")

    return {
        "timestamps": np.array(timestamps, dtype=float),
        "probs": np.array(unified_probs_list) if unified_probs_list else np.zeros((0, N_UNIFIED)),
        "has_audio": True,
    }


# ============================================================================
# 7. STREAMLIT UI
# (fusion, chart builders and overall_dominant_emotion now live in
#  utils/mer_core.py so the Upload and Live pages stay perfectly in sync.)
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
    sample_interval = st.slider("FER sampling interval (seconds)", 0.1, 3.0, 0.5, 0.1)
    w_fer, w_ser = FUSION_WEIGHTS
    st.caption(f"Fusion is fixed at {int(w_fer*100)}% FER / {int(w_ser*100)}% SER.")

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

        # Persist for the report pages (Detailed + Summary) and this page.
        store_results(fer_result, ser_result, fusion, source="Upload")
        st.session_state["analysis_done"] = True

    # ------------------------------------------------------------------
    # CONFIRMATION  (the full graphs now live in the report pages)
    # ------------------------------------------------------------------
    if st.session_state.get("analysis_done"):
        st.divider()
        st.success("✅ Analysis complete and saved.")

        results = st.session_state.get("mer_fusion")
        if results is not None:
            dom_label, dom_conf = overall_dominant_emotion(results["fused"])
            st.metric("Final Predicted Emotion (Fused)", dom_label.capitalize(), f"{dom_conf:.1%} confidence")

        st.info(
            "Open **📊 Summary Report** for the dominant-emotion overview and the "
            "distribution donut, or **🔍 Detailed Report** for the full interactive "
            "emotion timelines — both are in the sidebar."
        )
else:
    st.session_state["analysis_done"] = False