import streamlit as st
import cv2
import threading
import time
import re
import io
import json
import os
from datetime import datetime
from collections import Counter

import speech_recognition as sr
import whisper
from deepface import DeepFace

# ---------------- Page setup ----------------
st.set_page_config(page_title="Interview Analyzer", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #10141B; color: #EDEFF2; }
    .score-number { font-size: 56px; font-weight: 700; color: #E8A33D; line-height: 1; }
    .score-label { color: #8B95A1; font-size: 13px; }
    .badge { display:inline-block; padding: 4px 12px; border-radius: 999px;
             font-family: monospace; font-size: 12px; border: 1px solid #4FD1C5; color:#4FD1C5; }
    .report-card { background:#171D27; border:1px solid #262F3D; border-radius:16px; padding:28px; }
    div[data-testid="stMetricValue"] { color:#E8A33D; }
</style>
""", unsafe_allow_html=True)

FILLER_WORDS = {"um", "uh", "like", "you know", "actually", "basically", "literally"}
POSITIVE_EMOTIONS = {"neutral", "happy"}
NEGATIVE_EMOTIONS = {"fear", "sad", "angry", "disgust"}
EMOTION_ANALYSIS_EVERY_N_FRAMES = 8


# ---------------- Cached heavy resources (load once per server process) ----------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


whisper_model = load_whisper_model()
face_cascade = load_face_cascade()


# ---------------- Session state ----------------
if "data" not in st.session_state:
    st.session_state.data = {
        "full_transcript": [],   # [(timestamp, text)]
        "emotion_log": [],
        "latest_transcript": "Listening...",
        "session_active": True,
    }
if "audio_thread_started" not in st.session_state:
    st.session_state.audio_thread_started = False
if "report" not in st.session_state:
    st.session_state.report = None

data = st.session_state.data


# ---------------- Speech-to-text background thread ----------------
def transcribe_audio(shared):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            except sr.WaitTimeoutError:
                continue

        text = None
        try:
            text = recognizer.recognize_google(audio)
            shared["latest_transcript"] = f"[Online] {text}"
        except sr.UnknownValueError:
            shared["latest_transcript"] = "..."
            continue
        except sr.RequestError:
            try:
                audio_data = audio.get_wav_data()
                import soundfile as sf
                wav_io = io.BytesIO(audio_data)
                wav_data, samplerate = sf.read(wav_io)
                result = whisper_model.transcribe(wav_data.astype("float32"), fp16=False)
                text = result['text'].strip()
                shared["latest_transcript"] = f"[Offline] {text}"
            except Exception as e:
                shared["latest_transcript"] = f"Error: {e}"

        if text and shared["session_active"]:
            shared["full_transcript"].append((time.time(), text))


if not st.session_state.audio_thread_started:
    t = threading.Thread(target=transcribe_audio, args=(data,), daemon=True)
    t.start()
    st.session_state.audio_thread_started = True


# ---------------- Scoring ----------------
def compute_emotion_score(shared):
    log = shared["emotion_log"]
    if not log:
        return 50
    positive = sum(1 for e in log if e in POSITIVE_EMOTIONS)
    negative = sum(1 for e in log if e in NEGATIVE_EMOTIONS)
    total = len(log)
    score = ((positive - negative) / total + 1) / 2 * 100
    return round(max(0, min(100, score)), 1)


def compute_speech_score(shared):
    transcript = shared["full_transcript"]
    if not transcript:
        return 50
    all_words, filler_count = [], 0
    for _, text in transcript:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
        filler_count += sum(1 for w in words if w in FILLER_WORDS)
    total_words = len(all_words)
    if total_words == 0:
        return 50
    filler_ratio = filler_count / total_words
    return round(max(0, 100 - (filler_ratio * 500)), 1)


def compute_overall_score(shared):
    e = compute_emotion_score(shared)
    s = compute_speech_score(shared)
    return round((e + s) / 2, 1), e, s


def build_report(shared):
    overall, emo_score, speech_score = compute_overall_score(shared)
    log = shared["emotion_log"]
    transcript = shared["full_transcript"]

    if log:
        most_common_emotion, count = Counter(log).most_common(1)[0]
        ratio = count / len(log)
        emotion_summary = (f"Mostly {most_common_emotion.capitalize()}" if ratio > 0.6
                            else f"Mixed (mostly {most_common_emotion.capitalize()})")
    else:
        emotion_summary = "No data"

    all_words, filler_count = [], 0
    for _, text in transcript:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
        filler_count += sum(1 for w in words if w in FILLER_WORDS)
    total_words = len(all_words)
    filler_ratio = (filler_count / total_words) if total_words else 0

    if total_words == 0:
        speech_summary = "No speech detected"
    elif filler_ratio < 0.03:
        speech_summary = "Clear"
    elif filler_ratio < 0.08:
        speech_summary = "Somewhat clear"
    else:
        speech_summary = "Unclear (frequent filler words)"

    confidence_level = "High" if overall >= 75 else "Medium" if overall >= 50 else "Low"
    final_score_10 = round(overall / 10, 1)

    feedback_points = []
    if emo_score < 50:
        feedback_points.append("try to stay calmer and more composed on camera")
    if speech_score < 60:
        feedback_points.append("reduce filler words like 'um' and 'like' for clearer speech")
    if total_words < 20:
        feedback_points.append("speak more to give a fuller impression of your communication")
    if not log or len(log) < 10:
        feedback_points.append("ensure your face stays clearly visible to the camera")
    if not feedback_points:
        feedback_points.append("great job overall — keep maintaining eye contact and a steady tone")

    feedback = ("Improve " + "; ".join(feedback_points) + "."
                if len(feedback_points) > 1
                else feedback_points[0].capitalize() + ".")

    report_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "emotion_summary": emotion_summary,
        "speech_summary": speech_summary,
        "confidence_level": confidence_level,
        "final_score": final_score_10,
        "feedback": feedback,
        "raw_scores": {"overall": overall, "emotion_score": emo_score, "speech_score": speech_score},
    }

    os.makedirs("reports", exist_ok=True)
    filename = f"reports/interview_report_{report_data['timestamp']}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    report_data["saved_to"] = filename
    return report_data


# ---------------- Header ----------------
st.markdown("### 🎙️ Interview Analyzer")

col1, col2 = st.columns([2, 1])

with col1:
    run = st.checkbox("Start Camera", value=False)
    frame_placeholder = st.empty()
    transcript_placeholder = st.empty()

with col2:
    st.markdown("**Live Session**")
    score_placeholder = st.empty()
    emotion_meter_placeholder = st.empty()
    speech_meter_placeholder = st.empty()
    word_count_placeholder = st.empty()
    report_btn_placeholder = st.empty()

# ---------------- Camera loop (runs while checkbox is on) ----------------
if run:
    data["session_active"] = True
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open camera.")
    else:
        frame_count = 0
        current_emotion = "-"
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (232, 163, 61), 2)

                if frame_count % EMOTION_ANALYSIS_EVERY_N_FRAMES == 0:
                    face_roi = frame[y:y + h, x:x + w]
                    if face_roi.size != 0:
                        try:
                            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                            current_emotion = result[0]['dominant_emotion']
                            data["emotion_log"].append(current_emotion)
                        except Exception:
                            pass

                cv2.putText(frame, current_emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (232, 163, 61), 2)

            frame_placeholder.image(frame, channels="BGR")
            transcript_placeholder.markdown(f"**Transcript:** `{data['latest_transcript']}`")

            overall, emo_score, speech_score = compute_overall_score(data)
            word_count = sum(len(re.findall(r'\b\w+\b', t)) for _, t in data["full_transcript"])

            score_placeholder.markdown(
                f"<div class='score-number'>{overall}</div><div class='score-label'>/ 100</div>",
                unsafe_allow_html=True
            )
            emotion_meter_placeholder.progress(int(emo_score), text=f"Composure: {emo_score}")
            speech_meter_placeholder.progress(int(speech_score), text=f"Speech: {speech_score}")
            word_count_placeholder.markdown(f"**{word_count}** words captured")

            time.sleep(0.03)
            run = st.session_state.get("Start Camera", run)  # allow checkbox to interrupt loop

        cap.release()
else:
    data["session_active"] = False
    frame_placeholder.info("Camera is off. Check 'Start Camera' to begin.")

    if report_btn_placeholder.button("Generate Report", use_container_width=True):
        st.session_state.report = build_report(data)

# ---------------- Report card ----------------
if st.session_state.report:
    r = st.session_state.report
    st.markdown("---")
    with st.container():
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("**SESSION REPORT**")
        st.markdown(f"<div class='score-number'>{r['final_score']}</div><div class='score-label'>out of 10</div>",
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Emotion", r["emotion_summary"])
        c2.metric("Speech", r["speech_summary"])
        c3.metric("Confidence", r["confidence_level"])

        st.markdown(f"**Feedback:** {r['feedback']}")
        st.caption(f"Saved to {r['saved_to']}")

        if st.button("Start New Session"):
            st.session_state.data = {
                "full_transcript": [],
                "emotion_log": [],
                "latest_transcript": "Listening...",
                "session_active": True,
            }
            st.session_state.report = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)