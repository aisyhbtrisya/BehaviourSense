import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

@st.cache_resource
def load_ser_model():
    return load_model("individual_models/models/CNN_BiLSTM_RAVDESS.keras", compile=False)

model = load_ser_model()

emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad"
}

st.title("SER - Speech Emotion Recognition")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "m4a"]
)

def extract_ser_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050, duration=4.0)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=audio, sr=sr)
    )[:28]

    features = np.vstack([mfcc, chroma, mel])

    # Make sure shape becomes (80, 174)
    target_shape = (80, 174)

    if features.shape[1] < target_shape[1]:
        pad_width = target_shape[1] - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    else:
        features = features[:, :target_shape[1]]

    features = np.expand_dims(features, axis=-1)
    features = np.expand_dims(features, axis=0)

    return features

if uploaded_file is not None:
    st.audio(uploaded_file)

    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    features = extract_ser_features("temp_audio.wav")

    prediction = model.predict(features)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    emotion = emotion_labels[predicted_class]

    st.subheader("SER Result")
    st.write(f"**Predicted Emotion:** {emotion}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.subheader("Emotion Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{emotion_labels[i]}: {prob:.4f}")