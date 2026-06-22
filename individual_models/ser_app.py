import os
import tempfile
import librosa
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import AudioFileClip
import streamlit as st
from tensorflow.keras.models import load_model

# Adjust layout to wide mode so the two charts sit nicely side-by-side
st.set_page_config(layout="wide")


@st.cache_resource
def load_ser_model():
    return load_model("individual_models/models/CNN_BiLSTM_RAVDESS.keras", compile=False)


model = load_ser_model()

# Match the labels from your image
emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",  # Added surprise if your model supports it, adjust mapping if needed
}

st.title("SER - Speech Emotion Recognition")

uploaded_file = st.file_uploader(
    "Upload an audio or video file",
    type=["wav", "mp3", "m4a", "mp4", "avi", "mov", "mkv"],
)


def extract_ser_features(audio_window, sr):
    mfcc = librosa.feature.mfcc(y=audio_window, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio_window, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio_window, sr=sr, n_mels=28)
    mel_db = librosa.power_to_db(mel)

    features = np.vstack([mfcc, chroma, mel_db])
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
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension in ["mp4", "avi", "mov", "mkv"]:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file)

    with st.spinner("Processing media file and extracting audio..."):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file_extension}"
        ) as temp_input:
            temp_input.write(uploaded_file.read())
            temp_input_path = temp_input.name

        temp_wav_path = tempfile.mktemp(suffix=".wav")

        try:
            audio_clip = AudioFileClip(temp_input_path)
            audio_clip.write_audiofile(
                temp_wav_path, fps=22050, nbytes=2, codec="pcm_s16le", logger=None
            )
            audio_clip.close()
            audio, sr = librosa.load(temp_wav_path, sr=22050)
        finally:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

    # Windowing parameters
    window_size = 4
    overlap = 2
    step_size = window_size - overlap
    window_samples = window_size * sr
    step_samples = step_size * sr

    # Arrays to store timeline data for plotting
    time_stamps = []
    dominant_emotions = []
    confidence_scores = []
    all_probabilities = []

    for start in range(0, len(audio) - window_samples + 1, step_samples):
        end = start + window_samples
        audio_window = audio[start:end]

        features = extract_ser_features(audio_window, sr)
        prediction = model.predict(features)

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        # Center timestamp of the window for accurate graphing
        mid_time = (start + (window_samples / 2)) / sr

        time_stamps.append(mid_time)
        dominant_emotions.append(emotion_labels[predicted_class])
        confidence_scores.append(confidence)
        all_probabilities.append(prediction[0])

    # Check if we generated data points
    if len(time_stamps) == 0:
        st.error("Audio file is too short to parse into windows.")
    else:
        st.write("---")
        st.header("📊 Session Summary Analytics")

        # Create two columns to match your layout perfectly
        col1, col2 = st.columns(2)

        with col1:
            # Chart 1: Dominant Emotion Trend Over Session
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(
                time_stamps,
                confidence_scores,
                color="purple",
                marker="o",
                markersize=4,
                linewidth=2,
            )

            # Annotate text labels directly onto points
            for t, conf, emo in zip(
                time_stamps, confidence_scores, dominant_emotions
            ):
                ax1.text(
                    t,
                    conf + 0.02,
                    emo,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    alpha=0.8,
                )

            ax1.set_title("Dominant Emotion Trend Over Session", fontsize=10)
            ax1.set_xlabel("Time (Seconds)", fontsize=9)
            ax1.set_ylabel("Confidence Score", fontsize=9)
            ax1.set_ylim(-0.05, 1.1)
            ax1.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig1)

        with col2:
            # Chart 2: All Emotion Probabilities Breakdown
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            all_probabilities = np.array(all_probabilities)

            # Draw a line plot for each individual emotion track
            for idx, label in emotion_labels.items():
                # Verify that the index exists in the model's output slice
                if idx < all_probabilities.shape[1]:
                    ax2.plot(
                        time_stamps,
                        all_probabilities[:, idx],
                        label=label,
                        linewidth=1.5,
                    )

            ax2.set_title("All Emotion Probabilities Breakdown", fontsize=10)
            ax2.set_xlabel("Time (Seconds)", fontsize=9)
            ax2.set_ylabel("Probability", fontsize=9)
            ax2.set_ylim(-0.05, 1.1)
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right", fontsize=8)
            st.pyplot(fig2)

        # Overall summary metric blocks below the charts
        avg_probs = np.mean(all_probabilities, axis=0)
        final_class = np.argmax(avg_probs)

        st.write("---")
        m1, m2 = st.columns(2)
        m1.metric("Final Predicted Emotion", emotion_labels[final_class].upper())
        m2.metric("Final Session Confidence", f"{np.max(avg_probs):.2f}")