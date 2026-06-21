import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from tensorflow.keras.models import load_model

# 1. Load Model with caching so it stays fast
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

st.title("🎬 Video Emotion Analytics")
st.write("Upload a video file to analyze face emotions over time and plot analytics graphs.")

# Video Uploader Widget
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Streamlit passes a bytes object, OpenCV needs a file path. 
    # We save the uploaded bytes to a temporary local file.
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_video.read())
        video_path = tfile.name

    # Configuration Slider
    sample_interval = st.slider("Sampling Interval (Seconds)", min_value=0.1, max_value=3.0, value=0.5, step=0.1)

    if st.button("🚀 Start Video Analysis"):
        # Setup video capture
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip_interval = max(1, int(fps * sample_interval))
        
        # Lists for data collection
        timestamps = []
        dominant_emotions = []
        dominant_confidences = []
        emotion_trends = {emotion_labels[i]: [] for i in range(7)}
        
        frame_count = 0
        
        # Progress UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip_interval == 0:
                timestamp = round(frame_count / fps, 2)
                timestamps.append(timestamp)
                
                # Update progress bar text
                status_text.text(f"Processing time: {timestamp}s...")
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face = image_rgb[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (224, 224))
                    face_normalized = face_resized / 255.0
                    face_input = np.expand_dims(face_normalized, axis=0)
                    
                    prediction = model.predict(face_input, verbose=0)[0]
                    predicted_class = np.argmax(prediction)
                    
                    dominant_emotions.append(emotion_labels[predicted_class])
                    dominant_confidences.append(float(prediction[predicted_class]))
                    
                    for i in range(7):
                        emotion_trends[emotion_labels[i]].append(float(prediction[i]))
                else:
                    dominant_emotions.append("No Face")
                    dominant_confidences.append(0.0)
                    for i in range(7):
                        emotion_trends[emotion_labels[i]].append(0.0)

            frame_count += 1

        cap.release()
        status_text.text("Analysis finished! Generating graphs...")
        progress_bar.empty()

        # ------------------ PLOTTING THE GRAPHS IN STREAMLIT ------------------
        st.subheader("📊 Analytics Results")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Graph 1: Dominant Emotion Tracking Line
        ax1.plot(timestamps, dominant_confidences, color='purple', linewidth=2, marker='o', markersize=4)
        for i, txt in enumerate(dominant_emotions):
            if txt != "No Face" and i % 2 == 0:  # Annotate text labels to every other plot point
                ax1.annotate(txt, (timestamps[i], dominant_confidences[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        ax1.set_title("Dominant Emotion Trend")
        ax1.set_xlabel("Time (Seconds)")
        ax1.set_ylabel("Confidence Score")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_ylim(-0.05, 1.05)

        # Graph 2: Multiline breakdown of every emotion probability
        for emotion, values in emotion_trends.items():
            ax2.plot(timestamps, values, label=emotion, linewidth=1.5)
        ax2.set_title("All Emotion Probabilities Breakdown")
        ax2.set_xlabel("Time (Seconds)")
        ax2.set_ylabel("Probability")
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        # Display the matplotlib plot directly inside the web page
        st.pyplot(fig)