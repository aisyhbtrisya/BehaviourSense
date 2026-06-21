import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model

# 1. Load Model with caching
@st.cache_resource
def load_fer_model():
    return load_model("individual_models/models/MobileNet_Attention_RAFDB.h5", compile=False)

model = load_fer_model()

# RAF-DB labels
emotion_labels = {
    0: "surprise", 1: "fear", 2: "disgust", 3: "happy", 
    4: "sad", 5: "angry", 6: "neutral"
}

# Initialize Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("🎥 Live Webcam Emotion Analytics")
st.write("Click 'Start Webcam' to capture live emotions. Click 'Stop & Generate Graphs' to see your session analytics.")

# Initialize session state variables to hold data when the loop stops
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []
if "dominant_emotions" not in st.session_state:
    st.session_state.dominant_emotions = []
if "dominant_confidences" not in st.session_state:
    st.session_state.dominant_confidences = []
if "emotion_trends" not in st.session_state:
    st.session_state.emotion_trends = {emotion_labels[i]: [] for i in range(7)}

# Control Sliders & Buttons
sample_interval = st.slider("Analysis Interval (Seconds)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

col1, col2 = st.columns(2)
with col1:
    start_button = st.button("🟢 Start Webcam", use_container_width=True)
with col2:
    stop_button = st.button("🔴 Stop & Generate Graphs", use_container_width=True)

# Create layout placeholders for the live elements
frame_placeholder = st.empty()
metrics_placeholder = st.empty()

# --- WEBCAM LOOP ---
if start_button:
    # Reset history for a fresh session
    st.session_state.timestamps = []
    st.session_state.dominant_emotions = []
    st.session_state.dominant_confidences = []
    st.session_state.emotion_trends = {emotion_labels[i]: [] for i in range(7)}
    
    # Open default webcam (0)
    cap = cv2.VideoCapture(0)
    
    start_time = time.time()
    last_analysis_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam failed to start.")
            break

        # Mirror the frame for a more natural webcam feel
        frame = cv2.flip(frame, 1)
        
        current_time = time.time() - start_time
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect Face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Check if it's time to analyze based on your interval slider
        if current_time - last_analysis_time >= sample_interval:
            last_analysis_time = current_time
            
            # Store timestamp
            timestamp_rounded = round(current_time, 2)
            st.session_state.timestamps.append(timestamp_rounded)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = image_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (224, 224))
                face_normalized = face_resized / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)
                
                # Predict
                prediction = model.predict(face_input, verbose=0)[0]
                predicted_class = np.argmax(prediction)
                dominant_em = emotion_labels[predicted_class]
                conf = float(prediction[predicted_class])
                
                # Store records into session state
                st.session_state.dominant_emotions.append(dominant_em)
                st.session_state.dominant_confidences.append(conf)
                for i in range(7):
                    st.session_state.emotion_trends[emotion_labels[i]].append(float(prediction[i]))
            else:
                st.session_state.dominant_emotions.append("No Face")
                st.session_state.dominant_confidences.append(0.0)
                for i in range(7):
                    st.session_state.emotion_trends[emotion_labels[i]].append(0.0)

        # Draw box on screen dynamically (even between calculation frames)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if len(st.session_state.dominant_emotions) > 0:
                # Label the box with current live dominant emotion
                label = f"{st.session_state.dominant_emotions[-1]} ({st.session_state.dominant_confidences[-1]*100:.0f}%)"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 1. Update live camera feed frame placeholder
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        
        # 2. Update live probability readout percentages underneath camera
        if len(st.session_state.dominant_emotions) > 0:
            with metrics_placeholder.container():
                st.write(f"**Current Target Emotion:** {st.session_state.dominant_emotions[-1].upper()}")
                # Display a quick breakdown table or progress meters
                cols = st.columns(7)
                for idx, em_name in enumerate(emotion_labels.values()):
                    val = st.session_state.emotion_trends[em_name][-1]
                    cols[idx].metric(label=em_name, value=f"{val*100:.1f}%")

        # Allow Streamlit to yield processing to avoid browser hanging
        time.sleep(0.01)

    cap.release()

# --- PLOT FINAL SUMMARY HISTORIC CHARTS ---
# This executes either when the webcam breaks or if the session state contains data after stopping
if stop_button or (not start_button and len(st.session_state.timestamps) > 0):
    st.success("Webcam Session Stopped! Compiling graphs...")
    
    # Retrieve saved session history data
    ts = st.session_state.timestamps
    dom_em = st.session_state.dominant_emotions
    dom_conf = st.session_state.dominant_confidences
    trends = st.session_state.emotion_trends
    
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
        st.warning("No data tracked yet. Press 'Start Webcam' first.")