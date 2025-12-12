import streamlit as st
import cv2
import os
import av
import numpy as np
from collections import deque, Counter
import torch
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
from tensorflow.keras.models import load_model
import threading
import logging
import time

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
st.set_page_config(page_title="🚀 Project MAITRI", layout="wide")
st.title("🚀 Project MAITRI: Multimodal AI for Crew Wellness")

# Constants
VIDEO_FPS = 30  # Assuming a standard webcam FPS
ANALYSIS_INTERVAL_FRAMES = 15  # Analyze every 15 frames (~0.5 seconds at 30 FPS)
EMOTION_HISTORY_LENGTH = 15  # Keep history for smoothing
CONF_THRESHOLD = 0.4  # Increased for more confident predictions

# FER2013 label order (corrected)
EMOTION_LABELS = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'sad', 5: 'surprise', 6: 'neutral'
}

# Emotion hierarchy for fusion
EMOTION_HIERARCHY = {emo: 2 for emo in EMOTION_LABELS.values()}
EMOTION_HIERARCHY.update({"unknown": 1, "error": 1})

# Hubert audio label mapping
HUBERT_MAP = {
    "neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad",
    "fea": "fear", "dis": "disgust", "sur": "surprise"
}

# --- LOAD AI MODELS ---
@st.cache_resource
def load_models():
    try:
        # Force CPU usage
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("Loading video model...")
        video_model = load_model('emotion_model.h5')

        logger.info("Loading audio model...")
        model_path = "./models/superb/hubert-base-superb-er"
        if os.path.exists(model_path):
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            audio_model = HubertForSequenceClassification.from_pretrained(model_path)
        else:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")
            audio_model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-er")

        logger.info("Loading face cascade...")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        return video_model, feature_extractor, audio_model, face_cascade
    except Exception as e:
        logger.error(f"⚠️ Model load failed: {e}")
        st.error(f"⚠️ Model load failed: {e}. Please ensure you are connected to the internet or have the models downloaded locally.")
        st.stop()

video_model, audio_processor, audio_model, face_cascade = load_models()

# --- MULTIMODAL FUSION ---
def multimodal_fusion(video_emotion, audio_emotion, video_history, audio_history):
    video_mode = Counter(video_history).most_common(1)[0][0] if video_history else "unknown"
    audio_mode = Counter(audio_history).most_common(1)[0][0] if audio_history else "unknown"
    
    # Priority given to audio if it's not neutral
    if audio_mode != "neutral" and EMOTION_HIERARCHY.get(audio_mode, 1) > EMOTION_HIERARCHY.get(video_mode, 1):
        return audio_mode
    return video_mode

# --- VIDEO PROCESSOR ---
class MyVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.latest_video_frame = None
        
        self.video_history = deque(maxlen=EMOTION_HISTORY_LENGTH)
        self.audio_history = deque(maxlen=EMOTION_HISTORY_LENGTH)

        self.video_emotion = "neutral"
        self.audio_emotion = "neutral"
        self.fused_emotion = "neutral"
        self.critical_alert = False
        self.alert_start_time = None
        self.detected_face_region = None
        self.frame_lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        with self.frame_lock:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            # --- VIDEO ANALYSIS ---
            if self.frame_count % ANALYSIS_INTERVAL_FRAMES == 0:
                self.video_emotion = self._analyze_video(img)
            
            # --- AUDIO ANALYSIS ---
            # streamlit-webrtc now provides audio frames with the video frame
            if frame.audio:
                audio_frame = frame.audio[0]
                self.audio_emotion = self._analyze_audio(audio_frame)

            # --- FUSION AND UI OVERLAY ---
            self.fused_emotion = multimodal_fusion(self.video_emotion, self.audio_emotion, self.video_history, self.audio_history)
            
            # Update critical alert state
            if self.fused_emotion in ["angry", "disgust", "fear"]:
                if self.alert_start_time is None:
                    self.alert_start_time = time.time()
                elif time.time() - self.alert_start_time > 10:
                    self.critical_alert = True
            else:
                self.alert_start_time, self.critical_alert = None, False

            # Draw overlay on the frame
            img_with_overlay = self._draw_overlay(img, self.video_emotion, self.audio_emotion, self.fused_emotion, self.critical_alert)
            return av.VideoFrame.from_ndarray(img_with_overlay, format="bgr24")

    def _analyze_video(self, img):
        emotion = "unknown"
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                roi = gray[y:y+h, x:x+w]
                resized = cv2.resize(roi, (48, 48))
                preprocessed = np.expand_dims(np.expand_dims(resized, -1), 0) / 255.0
                preds = video_model.predict(preprocessed, verbose=0)[0]
                conf, pred_class = float(np.max(preds)), int(np.argmax(preds))
                emotion = EMOTION_LABELS[pred_class] if conf >= CONF_THRESHOLD else "unknown"
                self.detected_face_region = (x, y, w, h)
                logger.info(f"[VIDEO] Preds: {preds}, Conf: {conf:.2f}, Emotion: {emotion}")
        except Exception as e:
            emotion = "error"
            logger.error(f"Video analysis error: {e}")
        finally:
            self.video_history.append(emotion)
            return Counter(self.video_history).most_common(1)[0][0] if self.video_history else "unknown"

    def _analyze_audio(self, audio_frame: av.AudioFrame):
        emotion = "unknown"
        try:
            # Convert audio frame to numpy array
            audio = audio_frame.to_ndarray(format="fltp").reshape(-1)
            
            # Check for silence before processing
            if np.abs(audio).mean() < 0.001:  # Simple silence check
                emotion = "neutral"
            else:
                inputs = audio_processor(audio,
                                         sampling_rate=audio_frame.sample_rate,
                                         return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits = audio_model(**inputs).logits
                raw_label = audio_model.config.id2label[torch.argmax(logits).item()]
                emotion = HUBERT_MAP.get(raw_label, "unknown")
                logger.info(f"[AUDIO] Raw: {raw_label}, Mapped: {emotion}")
        except Exception as e:
            emotion = "unknown"
            logger.error(f"Audio analysis error: {e}")
        finally:
            self.audio_history.append(emotion)
            return Counter(self.audio_history).most_common(1)[0][0] if self.audio_history else "unknown"

    def _draw_overlay(self, img, vid_emo, aud_emo, fused_emo, is_critical):
        if self.detected_face_region:
            x, y, w, h = self.detected_face_region
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (380, 160), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        
        cv2.putText(img, f"Video: {vid_emo.capitalize()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Audio: {aud_emo.capitalize()}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        fused_text = f"Fused: {fused_emo.upper()}" + (" 🚨 ALERT" if is_critical else "")
        text_color = (0, 0, 255) if is_critical else (255, 255, 255)
        cv2.putText(img, fused_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        return img

# --- RESPONSE GENERATOR ---
def generate_response(emotion, is_critical):
    if is_critical: return "🚨 Critical state detected. Logging for ground control."
    if emotion in ["angry", "disgust", "fear"]: return "⚠️ High stress detected. Take a pause."
    if emotion in ["sad"]: return "It seems you're feeling low. I'm here with you."
    if emotion == "happy": return "😊 Great positivity! Keep going."
    if emotion == "surprise": return "😲 Something caught your attention!"
    return "✅ Systems stable."

# --- STREAMLIT UI ---
def main():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Live Astronaut Feed")
        webrtc_ctx = webrtc_streamer(
            key="maitri_stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=MyVideoProcessor,
            media_stream_constraints={"video": True, "audio": True},
            async_processing=False, # Use False for better synchronization
            rtc_configuration={"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]},
        )

    with col2:
        st.subheader("MAITRI Insights & Log")
        if "chat_history" not in st.session_state: st.session_state.chat_history = []
        if "processor" not in st.session_state: st.session_state.processor = None

        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            st.session_state.processor = webrtc_ctx.video_processor
            
            with st.session_state.processor.frame_lock:
                st.write(f"**Video:** `{st.session_state.processor.video_emotion}`")
                st.write(f"**Audio:** `{st.session_state.processor.audio_emotion}`")
                st.metric("SMOOTHED FUSED STATE", st.session_state.processor.fused_emotion.upper())
                if st.session_state.processor.critical_alert:
                    st.error("🚨 Critical emotional alert active!")

            if st.button("📝 Log Response"):
                with st.session_state.processor.frame_lock:
                    emotion, is_critical = st.session_state.processor.fused_emotion, st.session_state.processor.critical_alert
                response = generate_response(emotion, is_critical)
                st.session_state.chat_history.insert(0, {"emotion": emotion, "response": response})
                st.rerun()
        else:
            st.info("Click 'Start' to activate MAITRI.")

        st.write("---\n#### Response Log")
        for chat in st.session_state.chat_history:
            st.info(f"**[{chat['emotion'].upper()}]** {chat['response']}")

if __name__ == "__main__":
    main()