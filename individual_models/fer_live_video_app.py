import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ====================================================================
# 1. INITIALIZE SESSION STATE (Must be at the very top of your app!)
# ====================================================================
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = []

if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

# ====================================================================
# 2. DEFINE YOUR PROCESSOR
# ====================================================================
class EmotionProcessor(VideoProcessorBase):
    def __init__(self, data_list, interval):
        # Store a local reference to the list that was passed in
        self.data_list = data_list
        self.interval = interval

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Your emotion detection code here...
        # Instead of using st.session_state inside here, you use:
        # self.data_list.append(predicted_emotion)
        
        return frame.from_ndarray(img, format="bgr24")

# ====================================================================
# 3. WEBRTC STREAMER IMPLEMENTATION
# ====================================================================
st.title("🎥 WebRTC Live Emotion Analytics")

sample_interval = st.slider("Analysis Interval (Seconds)", 0.1, 2.0, 0.5)

# CRITICAL FIX: Pull the mutable list reference out into a local variable 
# BEFORE passing it into the lambda factory.
shared_analytics_list = st.session_state.analytics_data

ctx = webrtc_streamer(
    key="emotion-analysis",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    # Pass the local pointer variable, NOT st.session_state directly!
    video_processor_factory=lambda: EmotionProcessor(shared_analytics_list, sample_interval),
    media_stream_constraints={"video": True, "audio": False},
)