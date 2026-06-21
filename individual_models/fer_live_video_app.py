import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from tensorflow.keras.models import load_model

# ====================================================================
# 1. INITIALIZE DATA STRUCTURE & SESSION STATE
# ====================================================================
emotion_labels = {
    0: "surprise", 1: "fear", 2: "disgust", 3: "happy", 
    4: "sad", 5: "angry", 6: "neutral"
}

# We use a custom class to make sure background threads don't crash each other
class SharedAnalyticsData:
    def __init__(self):
        self.lock = threading.Lock()
        self.timestamps = []
        self.dominant_emotions = []
        self.dominant_confidences = []
        self.emotion_trends = {emotion_labels[i]: [] for i in range(7)}

# Initialize your state using the object structure instead of a plain list
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = SharedAnalyticsData()

# ====================================================================
# 2. LOAD MODEL & FACE DETECTOR
# ====================================================================
@st.cache_resource
def load_fer_model():
    return load_model("individual_models/models/MobileNet_Attention_RAFDB.h5", compile=False)

model = load_fer_model()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ====================================================================
# 3. DEFINE YOUR WEBRTC VIDEO PROCESSOR
# ====================================================================
class EmotionProcessor(VideoProcessorBase):
    def __init__(self, shared_data, interval):
        self.shared_data = shared_data
        self.interval = interval
        self.start_time = time.time()
        self.last_analysis_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror effect for natural webcam view
        
        current_time = time.time() - self.start_time
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Process emotions at your decided rate (sample_interval)
        if current_time - self.last_analysis_time >= self.interval:
            self.last_analysis_time = current_time
            
            dominant_em = "No Face"
            conf = 0.0
            preds = [0.0] * 7

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = image_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (224, 224))
                face_normalized = face_resized / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)
                
                # Model Predict
                prediction = model.predict(face_input, verbose=0)[0]
                predicted_class = np.argmax(prediction)
                dominant_em = emotion_labels[predicted_class]
                conf = float(prediction[predicted_class])
                preds = [float(p) for p in prediction]

            # Securely append to your shared data object using the thread lock
            with self.shared_data.lock:
                self.shared_data.timestamps.append(round(current_time, 2))
                self.shared_data.dominant_emotions.append(dominant_em)
                self.shared_data.dominant_confidences.append(conf)
                for i in range(7):
                    self.shared_data.emotion_trends[emotion_labels[i]].append(preds[i])

        # LIVE SHOWING: Draw the bounding boxes and text overlays onto the video stream frame
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Read the last captured emotion data to print above the green bounding box
            with self.shared_data.lock:
                if len(self.shared_data.dominant_emotions) > 0:
                    current_em = self.shared_data.dominant_emotions[-1]
                    current_conf = self.shared_data.dominant_confidences[-1]
                    if current_em != "No Face":
                        label = f"{current_em.upper()} ({current_conf * 100:.1f}%)"
                        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame.from_ndarray(img, format="bgr24")

# ====================================================================
# 4. WEBRTC STREAMER IMPLEMENTATION UI
# ====================================================================
st.title("🎥 WebRTC Live Emotion Analytics")

sample_interval = st.slider("Analysis Interval (Seconds)", 0.1, 2.0, 0.5)

# Pass the shared object directly to the factory
shared_analytics = st.session_state.analytics_data

ctx = webrtc_streamer(
    key="emotion-analysis",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_processor_factory=lambda: EmotionProcessor(shared_analytics, sample_interval),
    media_stream_constraints={"video": True, "audio": False},
)

# =====================================================================
# 5. ANALYTICS RENDERING MECHANISM
# =====================================================================
st.write("---")
if st.button("📊 Render Session Analytics Graphs", use_container_width=True):
    data = st.session_state.analytics_data
    
    # Thread-safely extract copies of lists for matplotlib to read safely
    with data.lock:
        ts = list(data.timestamps)
        dom_em = list(data.dominant_emotions)
        dom_conf = list(data.dominant_confidences)
        trends = {k: list(v) for k, v in data.emotion_trends.items()}

    if len(ts) > 0:
        st.subheader("📊 Session Summary Analytics")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Graph 1: Dominant Emotion Trend Line
        ax1.plot(ts, dom_conf, color='purple', linewidth=2, marker='o', markersize=4)
        for i, txt in enumerate(dom_em):
            if txt != "No Face" and i % 2 == 0:
                ax1.annotate(txt, (ts[i], dom_conf[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        ax1.set_title("Dominant Emotion Trend Over Session")
        ax1.set_xlabel("Time (Seconds)")
        ax1.set_ylabel("Confidence Score")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_ylim(-0.05, 1.05)

        # Graph 2: Multiline breakdown
        for emotion, values in trends.items():
            ax2.plot(ts, values, label=emotion, linewidth=1.5)
        ax2.set_title("All Emotion Probabilities Breakdown")
        ax2.set_xlabel("Time (Seconds)")
        ax2.set_ylabel("Probability")
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No data recorded yet. Please turn on your camera stream and use it before trying to generate analytics charts.")