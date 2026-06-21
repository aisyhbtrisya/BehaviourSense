import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

#Load Model
@st.cache_resource
def load_fer_model():
    return load_model("individual_models/models/MobileNet_Attention_RAFDB.h5", compile=False)

model = load_fer_model()

# RAF-DB labels
emotion_labels = {
    0: "surprise",
    1: "fear",
    2: "disgust",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "neutral"
}

# FER2013 labels
emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

st.title("FER - Facial Emotion Recognition")

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="Uploaded Image", use_container_width=True)

    # Detect face
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("No face detected. Try another clearer image.")
    else:
        x, y, w, h = faces[0]

        face = image_np[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        prediction = model.predict(face_input)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        emotion = emotion_labels[predicted_class]

        st.subheader("FER Result")
        st.write(f"**Predicted Emotion:** {emotion}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # Draw face box
        image_box = image_np.copy()
        cv2.rectangle(image_box, (x, y), (x+w, y+h), (0, 255, 0), 2)

        st.image(image_box, caption="Detected Face", use_container_width=True)

        st.subheader("Emotion Probabilities")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{emotion_labels[i]}: {prob:.4f}")