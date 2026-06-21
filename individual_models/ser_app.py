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

def extract_ser_features(audio_window, sr):
    mfcc = librosa.feature.mfcc(y=audio_window, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio_window, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio_window, sr=sr, n_mels=28)
    mel_db = librosa.power_to_db(mel)

    features = np.vstack([mfcc, chroma, mel_db])

    # Make sure shape becomes 80 x 174
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

    audio, sr = librosa.load(uploaded_file, sr=22050)

    window_size = 4
    overlap = 2
    step_size = window_size - overlap

    window_samples = window_size * sr
    step_samples = step_size * sr

    results = []

    for start in range(0, len(audio) - window_samples + 1, step_samples):
        end = start + window_samples
        audio_window = audio[start:end]

        features = extract_ser_features(audio_window, sr)

        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        emotion = emotion_labels[predicted_class]

        results.append({
            "start_time": start / sr,
            "end_time": end / sr,
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": prediction[0]
        })

    st.subheader("SER Result by Audio Window")

    if len(results) == 0:
        st.error("Audio is too short. Please upload audio longer than 4 seconds.")
    else:
        for i, result in enumerate(results):
            st.write(
                f"**Window {i+1}: "
                f"{result['start_time']:.1f}s - {result['end_time']:.1f}s**"
            )
            st.write(f"Predicted Emotion: **{result['emotion']}**")
            st.write(f"Confidence: **{result['confidence']:.2f}**")

            with st.expander("View emotion probabilities"):
                for j, prob in enumerate(result["probabilities"]):
                    st.write(f"{emotion_labels[j]}: {prob:.4f}")

        # Overall SER result
        avg_probs = np.mean([r["probabilities"] for r in results], axis=0)
        final_class = np.argmax(avg_probs)
        final_emotion = emotion_labels[final_class]
        final_confidence = np.max(avg_probs)

        st.subheader("Overall SER Result")
        st.write(f"**Final Emotion:** {final_emotion}")
        st.write(f"**Final Confidence:** {final_confidence:.2f}")