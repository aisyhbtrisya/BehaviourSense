"""
3_Live_Interview.py
====================
Live Multimodal Emotion Recognition (FER + SER).

Flow:
1. The webcam + microphone stream live via streamlit-webrtc.
2. A VIDEO processor runs Facial Emotion Recognition on sampled frames,
   draws a green box (+ label) around the detected face, and logs the
   per-emotion probabilities over time.
3. An AUDIO processor buffers microphone samples. While the stream plays,
   the main loop runs Speech Emotion Recognition on a rolling 2.5 s window
   and logs its per-emotion probabilities over time.
4. Two live charts update during recording: FER confidence and SER confidence.
5. When the user stops the stream, the page shows the SAME result UI as the
   Upload page — a Multimodal / FER / SER radio with interactive (zoom + hover)
   Plotly charts and the final fused emotion — by reusing utils/mer_core.

All heavy SER inference happens in the main Streamlit thread (inside the live
loop), NOT inside the webrtc audio callback, so the media stream stays smooth.
"""

import threading
import time

import cv2
import librosa
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    AudioProcessorBase,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

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
    warm_up_models,
    extract_ser_features,
    fuse_results,
    store_results,
    build_probability_chart,
    overall_dominant_emotion,
)

# set_page_config can only run once; guard it so the page works both standalone
# and when launched through app.py (which already calls it before runpy).
try:
    st.set_page_config(
        page_title="BehaviourSense AI",
        page_icon="brain",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

apply_theme()

# Rolling-window config for live SER (must match the training window length).
SER_WINDOW_SEC = 2.5     # length of each SER analysis window
SER_STEP_SEC = 1.0       # how often a new SER prediction is produced
SER_TARGET_SR = 22050    # SER models were trained at 22.05 kHz
AUDIO_KEEP_SEC = 6.0     # how much recent audio to keep buffered (bounds memory)
LOOP_SLEEP_SEC = 0.4     # live-chart refresh cadence


# ============================================================================
# THREAD-SAFE SHARED STORES
# These are plain objects (NOT st.session_state) so the webrtc worker threads
# can safely write to them via a lock. References are kept in session_state so
# the main thread can read them after the stream stops.
# ============================================================================
class FERStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.timestamps = []        # seconds since stream start
        self.unified_probs = []     # list[np.ndarray(N_UNIFIED)]
        self.face_found = []        # list[bool]
        self.last_label = "-"       # for the on-frame caption
        self.last_conf = 0.0
        self.session_start = time.time()


class FrameStore:
    """Holds only the most recent raw video frame (as a numpy array) so the
    callback thread can hand it off cheaply, and the main thread can pick it
    up and do the actual face-detection + FER prediction work."""
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_bgr = None     # most recent frame, BGR, already mirrored
        self.label_text = ""       # text to draw on the NEXT outgoing frame
        self.box = None            # (x, y, w, h) of the last detected face, or None


class AudioStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.samples = []           # list[np.ndarray] mono float32 at native sr
        self.sample_rate = None
        self.start_time = None


class SERStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.timestamps = []        # seconds since stream start
        self.unified_probs = []     # list[np.ndarray(N_UNIFIED)]


def _get_store(key, factory):
    if key not in st.session_state:
        st.session_state[key] = factory()
    return st.session_state[key]


# ============================================================================
# VIDEO PROCESSOR - thin frame relay ONLY.
#
# IMPORTANT: this callback runs on streamlit-webrtc's own aiortc event-loop
# thread, not Streamlit's main thread (this is documented behaviour of the
# library). Calling into TensorFlow or OpenCV's cascade classifier from that
# thread the first time those native libraries are touched is what was
# segfaulting the container with no Python traceback. So this callback does
# the absolute minimum: convert the frame, mirror it, draw whatever label the
# main thread already computed, and hand a copy of the raw frame to the main
# thread for the ACTUAL face detection + FER prediction. No cv2.Cascade, no
# model.predict(), and no other native ML call happens here.
# ============================================================================
class FERVideoProcessor(VideoProcessorBase):
    def __init__(self, frame_store, interval):
        self.frame_store = frame_store
        self.interval = interval
        self.start_time = time.time()
        self.last_handoff_time = -1e9

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # mirror for a natural selfie view
        except Exception:
            return frame

        current_time = time.time() - self.start_time

        # Hand off a copy of the frame to the main thread at most once per
        # `interval` seconds (cheap: just a numpy copy + lock, no detection).
        if current_time - self.last_handoff_time >= self.interval:
            self.last_handoff_time = current_time
            try:
                with self.frame_store.lock:
                    self.frame_store.latest_bgr = img.copy()
            except Exception:
                pass

        # Draw whatever label/box the main thread computed for the most
        # recent frame it processed (cheap: just text + a rectangle).
        try:
            with self.frame_store.lock:
                label_text = self.frame_store.label_text
                box = self.frame_store.box
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if label_text:
                    cv2.putText(img, label_text, (x, max(20, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception:
            pass

        try:
            return frame.from_ndarray(img, format="bgr24")
        except Exception:
            return frame


def maybe_run_live_fer(frame_store, fer_store, fer_model, face_cascade):
    """
    Runs ONCE PER MAIN-THREAD LOOP TICK (called from the same place as
    maybe_run_live_ser). Picks up the latest frame the callback handed off,
    does face detection (OpenCV) and FER prediction (TensorFlow) here, on the
    main thread, and writes both the emotion log AND the box/label back to
    frame_store so the callback can draw them on the next frame it sees.
    """
    with frame_store.lock:
        img = frame_store.latest_bgr
        frame_store.latest_bgr = None  # consume it; don't reprocess the same frame
    if img is None:
        return

    current_time = time.time()

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    except Exception:
        return

    if len(faces) == 0:
        with frame_store.lock:
            frame_store.box = None
            frame_store.label_text = ""
        with fer_store.lock:
            fer_store.timestamps.append(round(current_time - fer_store.session_start, 2))
            fer_store.unified_probs.append(np.zeros(N_UNIFIED))
            fer_store.face_found.append(False)
        return

    x, y, w, h = faces[0]
    face = image_rgb[y:y + h, x:x + w]
    if face.size == 0:
        with frame_store.lock:
            frame_store.box = (x, y, w, h)
            frame_store.label_text = ""
        return

    try:
        face_resized = cv2.resize(face, (224, 224))
        face_input = np.expand_dims(face_resized / 255.0, axis=0)
        prediction = fer_model.predict(face_input, verbose=0)[0]
    except Exception:
        with frame_store.lock:
            frame_store.box = (x, y, w, h)
        return

    unified = to_unified_vector(prediction, FER_LABELS)
    dom_idx = int(np.argmax(prediction))
    dom_name = FER_LABELS[dom_idx]
    dom_conf = float(prediction[dom_idx])

    with frame_store.lock:
        frame_store.box = (x, y, w, h)
        frame_store.label_text = f"{dom_name.upper()} ({dom_conf * 100:.1f}%)"

    with fer_store.lock:
        fer_store.timestamps.append(round(current_time - fer_store.session_start, 2))
        fer_store.unified_probs.append(unified)
        fer_store.face_found.append(True)
        fer_store.last_label = dom_name
        fer_store.last_conf = dom_conf


# ============================================================================
# AUDIO PROCESSOR - just buffers mic samples (cheap; no ML in the callback)
# ============================================================================
class AudioBufferProcessor(AudioProcessorBase):
    def __init__(self, store):
        self.store = store

    def recv(self, frame):
        try:
            raw = frame.to_ndarray()
            if np.issubdtype(raw.dtype, np.integer):
                denom = float(np.iinfo(raw.dtype).max)
                arr = raw.astype(np.float32) / denom
            else:
                arr = raw.astype(np.float32)
            if arr.ndim > 1:          # mix down channels -> mono
                arr = arr.mean(axis=0)

            with self.store.lock:
                if self.store.sample_rate is None:
                    self.store.sample_rate = int(frame.sample_rate)
                    self.store.start_time = time.time()
                self.store.samples.append(arr)
        except Exception:
            pass  # never let an audio hiccup kill the stream
        return frame  # pass audio through unchanged


# ============================================================================
# LIVE SER STEP - pulls the latest buffered audio, predicts, logs (main thread)
# ============================================================================
def maybe_run_live_ser(audio_store, ser_store, ser_model, scaler, last_run_holder):
    now = time.time()
    if now - last_run_holder[0] < SER_STEP_SEC:
        return

    with audio_store.lock:
        sr_native = audio_store.sample_rate
        start_time = audio_store.start_time
        if not sr_native or not audio_store.samples:
            return
        # Concatenate, trim to the last AUDIO_KEEP_SEC, and write the tail back
        # so memory stays bounded over a long interview.
        full = np.concatenate(audio_store.samples)
        keep = int(AUDIO_KEEP_SEC * sr_native)
        if len(full) > keep:
            full = full[-keep:]
        audio_store.samples = [full]

    need = int(SER_WINDOW_SEC * sr_native)
    if len(full) < need:
        return  # not enough audio for one full window yet

    window_native = full[-need:]
    try:
        window = librosa.resample(window_native, orig_sr=sr_native, target_sr=SER_TARGET_SR)
    except Exception:
        return

    target_len = int(SER_WINDOW_SEC * SER_TARGET_SR)
    if len(window) < target_len:
        window = np.pad(window, (0, target_len - len(window)))
    else:
        window = window[:target_len]

    try:
        feats = extract_ser_features(window, SER_TARGET_SR)
        scaled = scaler.transform(feats.reshape(1, -1))
        preds = ser_model.predict(scaled.reshape(1, -1, 1), verbose=0)[0]
    except Exception:
        return

    # Window-centre timestamp relative to stream start.
    t = max(0.0, round((now - start_time) - SER_WINDOW_SEC / 2.0, 2))
    with ser_store.lock:
        ser_store.timestamps.append(t)
        ser_store.unified_probs.append(to_unified_vector(preds, SER_LABELS))
    last_run_holder[0] = now


# ============================================================================
# HELPERS - build result dicts from the live stores (for fusion + charts)
# ============================================================================
def fer_result_from_store(store):
    with store.lock:
        ts = list(store.timestamps)
        probs = list(store.unified_probs)
        faces = list(store.face_found)
    return {
        "timestamps": np.array(ts, dtype=float),
        "probs": np.array(probs) if probs else np.zeros((0, N_UNIFIED)),
        "face_found": np.array(faces, dtype=bool),
    }


def ser_result_from_store(store):
    with store.lock:
        ts = list(store.timestamps)
        probs = list(store.unified_probs)
    return {
        "timestamps": np.array(ts, dtype=float),
        "probs": np.array(probs) if probs else np.zeros((0, N_UNIFIED)),
        "has_audio": len(probs) > 0,
    }


def live_line_chart(timestamps, probs_matrix, title):
    """Lightweight per-emotion line chart for the live (in-session) view."""
    return build_probability_chart(np.array(timestamps, dtype=float), probs_matrix, title)


# ============================================================================
# UI
# ============================================================================
st.title("Live Interview - Multimodal Emotion Recognition")
st.caption(
    "Your webcam and microphone stream live. A box is drawn around the detected "
    "face for FER, and speech is analysed on a rolling window for SER. Stop the "
    "stream to see the fused multimodal result."
)

fer_store = _get_store("live_fer_store", FERStore)
frame_store = _get_store("live_frame_store", FrameStore)
audio_store = _get_store("live_audio_store", AudioStore)
ser_store = _get_store("live_ser_store", SERStore)

# Load models (cached). Done up-front so the worker threads have them ready.
fer_model = load_fer_model()
ser_model, scaler, _encoder = load_ser_resources()
face_cascade = load_face_cascade()

# Run both models once on THIS (main) thread before the webrtc stream starts.
# See the docstring on warm_up_models for why this matters: it prevents the
# webrtc callback thread from ever being the first caller of a freshly loaded
# TF model, which is what was segfaulting the container.
if "models_warmed_up" not in st.session_state:
    with st.spinner("Warming up models..."):
        warm_up_models(fer_model, ser_model, scaler)
    st.session_state["models_warmed_up"] = True

st.subheader("Settings")
fer_interval = st.slider("FER sampling interval (seconds)", 0.1, 2.0, 0.5, 0.1)
w_fer, w_ser = FUSION_WEIGHTS
st.caption(f"Fusion is fixed at {int(w_fer*100)}% FER / {int(w_ser*100)}% SER.")

col_reset, _ = st.columns([1, 4])
with col_reset:
    if st.button("Clear session data"):
        for k in ("live_fer_store", "live_frame_store", "live_audio_store", "live_ser_store"):
            st.session_state.pop(k, None)
        st.session_state["live_show_results"] = False
        st.rerun()

st.divider()
st.subheader("Live stream")
st.caption("Click START to begin. Allow camera + microphone access when prompted. Click STOP when finished.")

ctx = webrtc_streamer(
    key="live-interview",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=lambda: FERVideoProcessor(frame_store, fer_interval),
    audio_processor_factory=lambda: AudioBufferProcessor(audio_store),
    media_stream_constraints={"video": True, "audio": True},
    async_processing=True,
)

# ------------------------------------------------------------------
# LIVE LOOP - runs only while the stream is playing.
# Updates the two live charts and produces rolling SER predictions.
# ------------------------------------------------------------------
if ctx.state.playing:
    st.info("Recording... live charts update below. Press STOP above to finish and view the fused result.")
    fer_metric_ph = st.empty()
    fer_chart_ph = st.empty()
    ser_chart_ph = st.empty()

    last_ser_run = [0.0]
    i = 0
    while ctx.state.playing:
        i += 1

        # ---- live FER (heavy work: cv2 cascade + TF predict — main thread only)
        maybe_run_live_fer(frame_store, fer_store, fer_model, face_cascade)

        # ---- live SER (heavy work, but on the main thread, not the callback)
        maybe_run_live_ser(audio_store, ser_store, ser_model, scaler, last_ser_run)

        # ---- live FER chart
        fer_res = fer_result_from_store(fer_store)
        if fer_res["probs"].shape[0] > 0:
            with fer_store.lock:
                lbl, conf = fer_store.last_label, fer_store.last_conf
            fer_metric_ph.metric("Current facial emotion", lbl.capitalize(), f"{conf:.1%}")
            fer_chart_ph.plotly_chart(
                live_line_chart(fer_res["timestamps"], fer_res["probs"], "Live FER - emotion confidence over time"),
                use_container_width=True,
                key=f"fer_live_{i}",
            )
        else:
            fer_metric_ph.info("Waiting for a face to appear in frame...")

        # ---- live SER chart
        ser_res = ser_result_from_store(ser_store)
        if ser_res["probs"].shape[0] > 0:
            ser_chart_ph.plotly_chart(
                live_line_chart(ser_res["timestamps"], ser_res["probs"], "Live SER - emotion confidence over time"),
                use_container_width=True,
                key=f"ser_live_{i}",
            )
        else:
            ser_chart_ph.info("Listening... first speech reading appears after ~2.5 s of audio.")

        time.sleep(LOOP_SLEEP_SEC)

    # Stream just stopped. Clear the live placeholders and fall through to the
    # confirmation below in the SAME run. (We deliberately avoid st.rerun() here:
    # rerunning re-enters the app.py navigation wrapper, and any hiccup in the
    # nav-radio state there can crash the page. Falling through is safe because
    # ctx.state.playing is now False.)
    fer_metric_ph.empty()
    fer_chart_ph.empty()
    ser_chart_ph.empty()

    # Build + persist the fused result so the report pages can display it.
    _fer_result = fer_result_from_store(fer_store)
    _ser_result = ser_result_from_store(ser_store)
    if _fer_result["probs"].shape[0] > 0 or _ser_result["probs"].shape[0] > 0:
        _fusion = fuse_results(_fer_result, _ser_result, w_fer, w_ser)
        store_results(_fer_result, _ser_result, _fusion, source="Live")
    st.session_state["live_show_results"] = True

# ------------------------------------------------------------------
# CONFIRMATION  (the full graphs now live in the report pages)
# ------------------------------------------------------------------
fer_result = fer_result_from_store(fer_store)
ser_result = ser_result_from_store(ser_store)
has_any_data = fer_result["probs"].shape[0] > 0 or ser_result["probs"].shape[0] > 0

if not ctx.state.playing and has_any_data:
    st.divider()
    st.success("✅ Live session finished and saved.")

    if fer_result["probs"].shape[0] > 0 and not fer_result["face_found"].any():
        st.warning("No face was detected in any sampled frame — FER results will be empty (zero confidence).")
    if ser_result["probs"].shape[0] == 0:
        st.warning("No speech was analysed (no or insufficient audio) — SER results are empty.")

    fusion = st.session_state.get("mer_fusion")
    if fusion is not None:
        dom_label, dom_conf = overall_dominant_emotion(fusion["fused"])
        st.metric("Final Predicted Emotion (Fused)", dom_label.capitalize(), f"{dom_conf:.1%} confidence")

    st.info(
        "Open **📊 Summary Report** for the dominant-emotion overview and the "
        "distribution donut, or **🔍 Detailed Report** for the full interactive "
        "emotion timelines — both are in the sidebar."
    )

elif not ctx.state.playing and not has_any_data:
    st.info("No session recorded yet. Start the stream above, let it run, then stop to see results.")