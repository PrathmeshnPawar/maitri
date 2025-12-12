import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import pipeline

# Load FER2013 model
video_model = load_model("emotion_model.h5")
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
CONF_THRESHOLD = 0.3  # Lower threshold so "unknown" happens less often

# Load HuggingFace audio model
audio_model = pipeline("audio-classification", model="superb/hubert-base-superb-er")

# WebRTC config
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.last_emotion = "unknown"
        self.last_confidences = {emo: 0.0 for emo in EMOTION_LABELS}

    def _process_video(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            # Predict
            predictions = video_model.predict(roi_gray, verbose=0)[0]
            confidence = float(np.max(predictions))
            predicted_class = int(np.argmax(predictions))
            top_emotion = EMOTION_LABELS[predicted_class]

            # Save for visualization
            self.last_confidences = {EMOTION_LABELS[i]: float(p) for i, p in enumerate(predictions)}

            # Apply threshold
            if confidence < CONF_THRESHOLD:
                self.last_emotion = "unknown"
            else:
                self.last_emotion = top_emotion

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return frame

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self._process_video(img)

        # Draw last detected emotion
        cv2.putText(img, f"Video Emotion: {self.last_emotion}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Draw confidence bars
        y0 = 80
        for i, (emo, conf) in enumerate(self.last_confidences.items()):
            bar_length = int(conf * 200)
            cv2.putText(img, f"{emo}: {conf:.2f}", (20, y0 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.rectangle(img, (150, y0 - 10 + i * 25),
                          (150 + bar_length, y0 + i * 25), (100, 255, 100), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit app
st.title("🎭 Real-time Emotion Recognition (Audio + Video)")
st.write("This demo shows video emotions with confidence bars and audio emotions separately.")

webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,   # ✅ use enum, not string
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": True},
)

# Audio processing
st.subheader("🎤 Audio Emotion Recognition")
st.info("Speak something to test emotion detection from audio")

uploaded_file = st.file_uploader("Upload a short audio file (.wav, .mp3)", type=["wav", "mp3"])
if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    results = audio_model("temp_audio.wav")
    st.write("Audio Emotion Results:", results)
