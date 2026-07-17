import cv2
import threading
import time
import re
from collections import Counter

import speech_recognition as sr
import whisper

from deepface import DeepFace

# ---------- Speech-to-text setup ----------
recognizer = sr.Recognizer()
mic = sr.Microphone()

whisper_model = whisper.load_model("base")

latest_transcript = "Listening..."
full_transcript = []  # stores every finalized sentence for scoring

FILLER_WORDS = {"um", "uh", "like", "you know", "actually", "basically", "literally"}


def transcribe_audio():
    global latest_transcript
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
            latest_transcript = f"[Online] {text}"
        except sr.UnknownValueError:
            latest_transcript = "..."
            continue
        except sr.RequestError:
            try:
                audio_data = audio.get_wav_data()
                import io
                import soundfile as sf

                wav_io = io.BytesIO(audio_data)
                data, samplerate = sf.read(wav_io)

                result = whisper_model.transcribe(data.astype("float32"), fp16=False)
                text = result['text'].strip()
                latest_transcript = f"[Offline] {text}"
            except Exception as e:
                latest_transcript = f"Error: {e}"

        if text:
            full_transcript.append((time.time(), text))


# Start audio thread
audio_thread = threading.Thread(target=transcribe_audio, daemon=True)
audio_thread.start()

# ---------- Emotion tracking ----------
emotion_log = []  # store every detected emotion for scoring

POSITIVE_EMOTIONS = {"neutral", "happy"}
NEGATIVE_EMOTIONS = {"fear", "sad", "angry", "disgust"}


def compute_emotion_score():
    if not emotion_log:
        return 50  # neutral default if no data yet
    positive = sum(1 for e in emotion_log if e in POSITIVE_EMOTIONS)
    negative = sum(1 for e in emotion_log if e in NEGATIVE_EMOTIONS)
    total = len(emotion_log)
    # scale 0-100: more positive emotions -> higher score
    score = ((positive - negative) / total + 1) / 2 * 100
    return round(max(0, min(100, score)), 1)


def compute_speech_score():
    if not full_transcript:
        return 50
    all_words = []
    filler_count = 0
    for _, text in full_transcript:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
        for w in words:
            if w in FILLER_WORDS:
                filler_count += 1

    total_words = len(all_words)
    if total_words == 0:
        return 50

    filler_ratio = filler_count / total_words
    # fewer fillers -> higher score. 0% fillers = 100, 20%+ fillers = 0
    score = max(0, 100 - (filler_ratio * 500))
    return round(score, 1)


def compute_overall_score():
    e = compute_emotion_score()
    s = compute_speech_score()
    return round((e + s) / 2, 1), e, s


# ---------- Video + face + emotion setup ----------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_roi = frame[y:y + h, x:x + w]

        if face_roi.size != 0:
            try:
                result = DeepFace.analyze(
                    face_roi,
                    actions=['emotion'],
                    enforce_detection=False
                )
                emotion = result[0]['dominant_emotion']
                emotion_log.append(emotion)
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception:
                pass

    # Overlay live transcript
    cv2.putText(frame, latest_transcript, (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Overlay live score
    overall, emo_score, speech_score = compute_overall_score()
    score_text = f"Score: {overall}  (Emotion: {emo_score} | Speech: {speech_score})"
    cv2.putText(frame, score_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    cv2.imshow("Interview Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------- Final summary report ----------
overall, emo_score, speech_score = compute_overall_score()

# Dominant emotion label
if emotion_log:
    most_common_emotion, count = Counter(emotion_log).most_common(1)[0]
    emotion_ratio = count / len(emotion_log)
    if emotion_ratio > 0.6:
        emotion_summary = f"Mostly {most_common_emotion.capitalize()}"
    else:
        emotion_summary = f"Mixed (mostly {most_common_emotion.capitalize()})"
else:
    emotion_summary = "No data"

# Speech clarity label
all_words = []
filler_count = 0
for _, text in full_transcript:
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

# Confidence level (based on overall score)
if overall >= 75:
    confidence_level = "High"
elif overall >= 50:
    confidence_level = "Medium"
else:
    confidence_level = "Low"

# Final score out of 10
final_score_10 = round(overall / 10, 1)

# Feedback generation
feedback_points = []

if emo_score < 50:
    feedback_points.append("try to stay calmer and more composed on camera")
if speech_score < 60:
    feedback_points.append("reduce filler words like 'um' and 'like' for clearer speech")
if total_words < 20:
    feedback_points.append("speak more to give a fuller impression of your communication")
if not emotion_log or len(emotion_log) < 10:
    feedback_points.append("ensure your face stays clearly visible to the camera")
if not feedback_points:
    feedback_points.append("great job overall — keep maintaining eye contact and a steady tone")

feedback = "Improve " + "; ".join(feedback_points) + "." if len(feedback_points) > 1 else feedback_points[0].capitalize() + "."

# ---------- Save results to file ----------
import json
import os
from datetime import datetime

os.makedirs("reports", exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_data = {
    "timestamp": timestamp,
    "emotion_summary": emotion_summary,
    "speech_summary": speech_summary,
    "confidence_level": confidence_level,
    "final_score": final_score_10,
    "feedback": feedback,
    "raw_scores": {
        "overall": overall,
        "emotion_score": emo_score,
        "speech_score": speech_score
    }
}

filename = f"reports/interview_report_{timestamp}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=4)

print("\nExample output:")
print(f"Emotion: {emotion_summary}")
print(f"Speech: {speech_summary}")
print(f"Confidence: {confidence_level}")
print(f"Final Score: {final_score_10}/10")
print(f"Feedback: {feedback}")
print(f"\nSaved report to: {filename}")