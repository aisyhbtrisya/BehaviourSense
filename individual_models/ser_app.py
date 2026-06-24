import os
import librosa
import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

# 1. Configuration matching the training notebook
TARGET_SR = 22050
DURATION = 2.5
OFFSET = 0.6
FRAME_LENGTH = 2048
HOP_LENGTH = 512
ALPHA = 0.035
BETA = 0.7

# 2. Paths - Update these to match your actual file locations
MODEL_PATH = "path_to_your/SER_model.keras"  # Update this
SCALER_PATH = "path_to_your/scaler.pkl"     # Update this

# 3. Load model and scaler
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH, compile=False)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_resources()

# Emotion labels based on CREMA-D mapping
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad"
}

def extract_features_from_signal(y, sr):
    zcr = np.squeeze(librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH))
    rmse = np.squeeze(librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH).flatten()
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH).flatten()
    return np.concatenate((zcr, rmse, mfcc, chroma))

def process_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=TARGET_SR, duration=DURATION, offset=OFFSET)
    required_samples = int(TARGET_SR * DURATION)
    if len(y) < required_samples:
        y = np.pad(y, (0, required_samples - len(y)), mode='constant')

    f_oa = extract_features_from_signal(y, sr)
    f_na = extract_features_from_signal(y + ALPHA * np.random.normal(0, 1, len(y)), sr)
    f_pa = extract_features_from_signal(librosa.effects.pitch_shift(y, sr=sr, n_steps=BETA), sr)
    f_comb = extract_features_from_signal(librosa.effects.pitch_shift(y + ALPHA * np.random.normal(0, 1, len(y)), sr=sr, n_steps=BETA), sr)
    
    return np.concatenate((f_oa, f_na, f_pa, f_comb))

st.title("SER - Speech Emotion Recognition (CREMA-D)")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    # Save to temp file for processing
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process
    features = process_audio_file("temp.wav")
    features = np.nan_to_num(features, nan=0.0)
    
    # Scale and reshape
    features = scaler.transform(features.reshape(1, -1))
    features = np.expand_dims(features, axis=-1)
    
    # Predict
    preds = model.predict(features)
    predicted_emotion = emotion_labels[np.argmax(preds)]
    
    st.write(f"### Predicted Emotion: {predicted_emotion}")
