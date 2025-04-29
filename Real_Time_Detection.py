import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Function to extract frames from a video
def extract_frames(video_path, output_size=(299, 299), frame_count=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < frame_count:
        st.warning(f"Only {total_frames} frames available in: {video_path}")

    step = max(total_frames // frame_count, 1)

    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, output_size)
        frame = frame.astype('float32') / 255.0
        frames.append(frame)

    cap.release()

    while len(frames) < frame_count:
        frames.append(np.zeros((output_size[0], output_size[1], 3), dtype=np.float32))

    return np.array(frames)

# Prediction function
def predict_video(video_path, model, output_size=(299, 299), frame_count=10):
    frames = extract_frames(video_path, output_size, frame_count)
    frames = np.expand_dims(frames, axis=0)  # Shape: (1, 10, 299, 299, 3)
    prediction = model.predict(frames)
    label = "FAKE" if np.argmax(prediction) == 1 else "REAL"
    confidence = prediction[0][np.argmax(prediction)]
    return label, confidence

# Streamlit app for video upload and prediction
def main():
    st.title("Deepfake AI Video detection 2025")

    # Load your trained model
    model_path = r"C:\Users\kumar\Downloads\deepfake_detection_model_final.keras"
    model = load_model(model_path)

    # File uploader
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4"])

    if uploaded_video is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(temp_video_path)

        label, confidence = predict_video(temp_video_path, model)
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}")

        os.remove(temp_video_path)

if __name__ == "__main__":
    main()
