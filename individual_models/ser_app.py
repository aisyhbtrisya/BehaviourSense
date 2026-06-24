import os
import tempfile
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import streamlit as st
from moviepy.editor import AudioFileClip
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide")

# --- 1. Resource Loading ---
@st.cache_resource
def load_resources():
    # Paths updated to your models folder
    model_path = "individual_models/models/SER_CNN_BiLSTM_CREMAD.keras"
    scaler_path = "individual_models/models/scaler.pkl"
    # Loading encoder just in case, though we use the dictionary for mapping
    encoder_path = "individual_models/models/encoder.pkl"
    
    model = load_model(model_path, compile=False)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
        
    return model, scaler, encoder

model, scaler, encoder = load_resources()

# CREMA-D Labels
emotion_labels = {
    0: "Angry", 1: "Disgust", 2: "Fear", 
    3: "Happy", 4: "Neutral", 5: "Sad"
}

st.title("SER - Speech Emotion Recognition (CREMA-D)")

# --- 2. Feature Extraction Pipeline (Matches Training) ---
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
    f_pa = get_features(librosa.effects.pitch_shift(y, sr=sr, n_steps=BETA))
    f_comb = get_features(librosa.effects.pitch_shift(y + ALPHA * np.random.normal(0, 1, len(y)), sr=sr, n_steps=BETA))
    
    return np.nan_to_num(np.concatenate((f_oa, f_na, f_pa, f_comb)), nan=0.0)

# --- 3. UI and Processing ---
uploaded_file = st.file_uploader("Upload audio/video", type=["wav", "mp3", "mp4", "mov", "mkv"])

if uploaded_file:
    # Handle media extraction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded_file.read())
        tmp_path = tmp_in.name
    
    temp_wav = tempfile.mktemp(suffix=".wav")
    clip = AudioFileClip(tmp_path)
    clip.write_audiofile(temp_wav, fps=22050, logger=None)
    audio, sr = librosa.load(temp_wav, sr=22050)
    
    # Sliding window processing (2.5s to match training)
    window_samples = int(2.5 * sr)
    step_samples = int(1.0 * sr) # Slide every 1 second
    
    all_probabilities, time_stamps, dominant_emotions, confidences = [], [], [], []

    for start in range(0, len(audio) - window_samples + 1, step_samples):
        window = audio[start : start + window_samples]
        features = extract_ser_features(window, sr)
        
        # Scale and Predict
        scaled_features = scaler.transform(features.reshape(1, -1))
        preds = model.predict(scaled_features.reshape(1, -1, 1), verbose=0)
        
        all_probabilities.append(preds[0])
        dominant_emotions.append(emotion_labels[np.argmax(preds)])
        confidences.append(np.max(preds))
        time_stamps.append((start + window_samples/2) / sr)

    # --- 4. Graphs ---
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(time_stamps, confidences, marker='o', color='purple')
        ax1.set_title("Dominant Emotion Trend")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        all_probs = np.array(all_probabilities)
        for idx, label in emotion_labels.items():
            ax2.plot(time_stamps, all_probs[:, idx], label=label)
        ax2.legend()
        ax2.set_title("Probability Breakdown")
        st.pyplot(fig2)

    st.metric("Final Predicted Emotion", emotion_labels[np.argmax(np.mean(all_probs, axis=0))])
