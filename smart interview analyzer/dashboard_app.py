import streamlit as st
import cv2
import threading
import time
import re
import io
import json
import os
import glob
from datetime import datetime
from collections import Counter

import plotly.graph_objects as go
import speech_recognition as sr
import whisper
from deepface import DeepFace

# ============================================================
# PAGE CONFIG + GLOBAL STYLE
# ============================================================
st.set_page_config(page_title="AI Interview Analyzer", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .card-title { font-family: 'Poppins', sans-serif; }

.stApp { background-color: #0B0F1C; color: #E7E9F3; }
section[data-testid="stSidebar"] { background-color: #10152A; border-right: 1px solid #1E2540; }

/* Card container */
.card {
    background: #131A30;
    border: 1px solid #232B4A;
    border-radius: 16px;
    padding: 20px 22px;
    height: 100%;
}
.card-title {
    font-size: 15px;
    font-weight: 600;
    color: #E7E9F3;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.card-sub { color: #7C86A8; font-size: 13px; margin-bottom: 10px; }

/* Badges */
.badge-active {
    background: rgba(46, 204, 113, 0.12);
    color: #2ECC71;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 600;
}
.badge-dot { height:7px; width:7px; border-radius:50%; background:#2ECC71; display:inline-block; margin-right:6px; }

/* Progress bars */
.pbar-track { background: #0B0F1C; border-radius: 999px; height: 8px; overflow: hidden; margin: 8px 0 4px 0; }
.pbar-fill { height: 100%; border-radius: 999px; }

/* Metric value big */
.metric-value { font-size: 28px; font-weight: 700; color: #FFFFFF; }
.metric-label { color: #7C86A8; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }

/* Feedback list */
.fb-item { display:flex; align-items:flex-start; gap:10px; font-size:14px; margin-bottom:10px; color:#D6DAEF; }
.fb-good { color:#2ECC71; }
.fb-warn { color:#F5A623; }

/* Gauge label */
.gauge-score { text-align:center; font-size:44px; font-weight:700; color:#fff; margin-top:-10px;}
.gauge-outof { text-align:center; color:#7C86A8; font-size:13px; }
.gauge-tag { text-align:center; font-weight:600; margin-top:6px; }

/* Sidebar nav */
.nav-item {
    display:flex; align-items:center; gap:10px;
    padding: 10px 14px; border-radius: 10px; margin-bottom: 4px;
    color: #AEB6D6; font-size: 14px; font-weight: 500;
}
.nav-item-active {
    background: linear-gradient(90deg, #6C5CE7, #4C6FFF);
    color: white;
}

button[kind="primary"] {
    background: linear-gradient(90deg, #6C5CE7, #4C6FFF) !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

FILLER_WORDS = {"um", "uh", "like", "you know", "actually", "basically", "literally"}
POSITIVE_EMOTIONS = {"neutral", "happy"}
NEGATIVE_EMOTIONS = {"fear", "sad", "angry", "disgust"}
KEYWORD_BANK = {"experience", "team", "project", "result", "achieved", "learned",
                 "challenge", "goal", "responsible", "collaborate", "solved", "impact"}
EMOTION_ANALYSIS_EVERY_N_FRAMES = 8


# ============================================================
# CACHED RESOURCES
# ============================================================
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


whisper_model = load_whisper_model()
face_cascade = load_face_cascade()

# ============================================================
# SESSION STATE
# ============================================================
if "data" not in st.session_state:
    st.session_state.data = {
        "full_transcript": [],
        "emotion_log": [],
        "latest_transcript": "Listening...",
        "session_active": True,
        "score_trail": [],       # rolling overall-score samples for sparklines
        "start_time": time.time(),
    }
if "audio_thread_started" not in st.session_state:
    st.session_state.audio_thread_started = False
if "running" not in st.session_state:
    st.session_state.running = False

data = st.session_state.data


# ============================================================
# BACKGROUND SPEECH THREAD
# ============================================================
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


# ============================================================
# SCORING
# ============================================================
def compute_emotion_score(shared):
    log = shared["emotion_log"]
    if not log:
        return 50.0
    positive = sum(1 for e in log if e in POSITIVE_EMOTIONS)
    negative = sum(1 for e in log if e in NEGATIVE_EMOTIONS)
    total = len(log)
    score = ((positive - negative) / total + 1) / 2 * 100
    return round(max(0, min(100, score)), 1)


def compute_speech_score(shared):
    transcript = shared["full_transcript"]
    if not transcript:
        return 50.0
    all_words, filler_count = [], 0
    for _, text in transcript:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
        filler_count += sum(1 for w in words if w in FILLER_WORDS)
    total_words = len(all_words)
    if total_words == 0:
        return 50.0
    filler_ratio = filler_count / total_words
    return round(max(0, 100 - (filler_ratio * 500)), 1)


def compute_keyword_score(shared):
    transcript = shared["full_transcript"]
    all_words = []
    for _, text in transcript:
        all_words.extend(re.findall(r'\b\w+\b', text.lower()))
    if not all_words:
        return 50.0
    hits = sum(1 for w in all_words if w in KEYWORD_BANK)
    ratio = hits / max(len(all_words), 1)
    return round(min(100, 50 + ratio * 800), 1)


def compute_overall(shared):
    e = compute_emotion_score(shared)
    s = compute_speech_score(shared)
    return round((e + s) / 2, 1), e, s


def emotion_summary_label(shared):
    log = shared["emotion_log"]
    if not log:
        return "No data", "-"
    most_common, count = Counter(log).most_common(1)[0]
    ratio = count / len(log)
    label = f"Mostly {most_common.capitalize()}" if ratio > 0.6 else f"Mixed ({most_common.capitalize()})"
    return label, most_common.capitalize()


def speech_summary_label(shared):
    transcript = shared["full_transcript"]
    all_words, filler_count = [], 0
    for _, text in transcript:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
        filler_count += sum(1 for w in words if w in FILLER_WORDS)
    total_words = len(all_words)
    if total_words == 0:
        return "No speech yet"
    ratio = filler_count / total_words
    if ratio < 0.03:
        return "Clear"
    elif ratio < 0.08:
        return "Somewhat clear"
    return "Unclear"


def build_feedback(shared, emo_score, speech_score):
    points = []
    if emo_score >= 60:
        points.append(("good", "Maintained a calm, composed expression."))
    else:
        points.append(("warn", "Work on staying calmer and more composed on camera."))

    if speech_score >= 70:
        points.append(("good", "Your speech is clear and understandable."))
    else:
        points.append(("warn", "Reduce filler words like 'um', 'uh', 'like'."))

    word_count = sum(len(re.findall(r'\b\w+\b', t)) for _, t in shared["full_transcript"])
    if word_count >= 40:
        points.append(("good", "Good amount of detail in your responses."))
    else:
        points.append(("warn", "Try to speak more to fully showcase your communication."))

    if len(shared["emotion_log"]) < 10:
        points.append(("warn", "Ensure your face stays clearly visible to the camera."))
    else:
        points.append(("good", "Maintained consistent visibility to the camera."))

    return points


def save_report(shared):
    overall, emo_score, speech_score = compute_overall(shared)
    emo_label, dominant = emotion_summary_label(shared)
    speech_label = speech_summary_label(shared)
    confidence = "High" if overall >= 75 else "Medium" if overall >= 50 else "Low"
    final_10 = round(overall / 10, 1)

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "emotion_summary": emo_label,
        "speech_summary": speech_label,
        "confidence_level": confidence,
        "final_score": final_10,
        "raw_scores": {"overall": overall, "emotion_score": emo_score, "speech_score": speech_score},
    }
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/interview_report_{report['timestamp']}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    report["saved_to"] = filename
    return report


def load_score_history():
    files = sorted(glob.glob("reports/interview_report_*.json"))
    scores = []
    for f in files[-5:]:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                r = json.load(fh)
                scores.append(r.get("final_score", 0))
        except Exception:
            continue
    return scores


# ============================================================
# CHART HELPERS
# ============================================================
def gauge_chart(score_10):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_10,
        number={"font": {"size": 1, "color": "rgba(0,0,0,0)"}},  # hide built-in number, we render our own
        gauge={
            "axis": {"range": [0, 10], "tickcolor": "#232B4A", "tickfont": {"color": "#7C86A8"}},
            "bar": {"color": "rgba(0,0,0,0)"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 4], "color": "#E8615D"},
                {"range": [4, 7], "color": "#F5A623"},
                {"range": [7, 10], "color": "#2ECC71"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.85,
                "value": score_10,
            },
        },
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E7E9F3"},
    )
    return fig


def sparkline(values, color):
    if not values:
        values = [50, 50]
    fig = go.Figure(go.Scatter(y=values, mode="lines", line=dict(color=color, width=2),
                                fill="tozeroy", fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba")))
    fig.update_layout(
        height=50, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, 100]),
        showlegend=False,
    )
    return fig


def history_chart(history_scores, current_score):
    labels = [f"Session {i+1}" for i in range(len(history_scores))] + ["Current"]
    values = history_scores + [current_score]
    fig = go.Figure(go.Scatter(
        x=labels, y=values, mode="lines+markers",
        line=dict(color="#8B7CF6", width=3), marker=dict(size=7, color="#8B7CF6"),
        fill="tozeroy", fillcolor="rgba(139,124,246,0.12)",
    ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#7C86A8", gridcolor="#1E2540"),
        yaxis=dict(color="#7C86A8", gridcolor="#1E2540", range=[0, 10]),
        font={"color": "#E7E9F3"},
    )
    return fig


def progress_bar_html(pct, color):
    return f"""<div class="pbar-track"><div class="pbar-fill" style="width:{pct}%; background:{color};"></div></div>"""


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🧠 AI Interview\n**Analyzer**")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="nav-item nav-item-active">🏠&nbsp;&nbsp;Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📷&nbsp;&nbsp;Live Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📊&nbsp;&nbsp;Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">🕘&nbsp;&nbsp;History</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">⚙️&nbsp;&nbsp;Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">ℹ️&nbsp;&nbsp;About</div>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="card"><b>Get AI-powered feedback</b><br>'
        '<span style="color:#7C86A8; font-size:13px;">on your interview performance in real-time.</span></div>',
        unsafe_allow_html=True
    )

# ============================================================
# HEADER
# ============================================================
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("## Interview Analysis Dashboard")
    st.markdown('<span style="color:#7C86A8;">Real-time analysis of your interview performance</span>',
                unsafe_allow_html=True)
with h2:
    st.write("")
    start_stop = st.toggle("▶ Start / Stop Analysis", value=st.session_state.running)
    st.session_state.running = start_stop

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# ROW 1: Video | Gauge | Side stats
# ============================================================
col_video, col_gauge, col_stats = st.columns([1.4, 1, 1])

with col_video:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    top = st.columns([2, 1])
    top[0].markdown('<div class="card-title">Live Video Feed</div>', unsafe_allow_html=True)
    badge = top[1]
    frame_placeholder = st.empty()
    meta_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with col_gauge:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Overall Performance</div>', unsafe_allow_html=True)
    gauge_placeholder = st.empty()
    gauge_text_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with col_stats:
    conf_card = st.empty()
    speech_card = st.empty()
    emo_card = st.empty()


def render_side_cards(overall, emo_score, speech_score, emotion_label):
    confidence = "High" if overall >= 75 else "Medium" if overall >= 50 else "Low"
    conf_card.markdown(f"""
        <div class="card" style="margin-bottom:12px;">
            <div class="card-title">🛡️ Confidence Level</div>
            <div class="metric-value">{overall:.0f}%</div>
            <div class="metric-label">{confidence}</div>
        </div>
    """, unsafe_allow_html=True)
    speech_card.markdown(f"""
        <div class="card" style="margin-bottom:12px;">
            <div class="card-title">🎤 Speech Clarity</div>
            <div class="metric-value">{speech_score:.0f}%</div>
            <div class="metric-label">{speech_summary_label(data)}</div>
        </div>
    """, unsafe_allow_html=True)
    emo_card.markdown(f"""
        <div class="card">
            <div class="card-title">🙂 Emotional State</div>
            <div class="metric-value">{emotion_label}</div>
            <div class="metric-label">{emotion_summary_label(data)[0]}</div>
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# ROW 2: Detailed Analysis
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Detailed Analysis")
d1, d2, d3, d4 = st.columns(4)
detail_placeholders = [d1.empty(), d2.empty(), d3.empty(), d4.empty()]

# ============================================================
# ROW 3: Feedback | Score history | Final score
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
f1, f2, f3 = st.columns([1.1, 1.4, 0.8])
feedback_placeholder = f1.empty()
history_placeholder = f2.empty()
final_placeholder = f3.empty()


def render_detailed_cards(emo_score, speech_score, overall, keyword_score):
    emo_label, _ = emotion_summary_label(data)
    detail_placeholders[0].markdown(f"""
        <div class="card">
            <div class="card-title">🙂 Emotion Detection</div>
            <div style="font-weight:600;">{emo_label}</div>
            {progress_bar_html(emo_score, "#8B5CF6")}
            <span style="color:#7C86A8; font-size:12px;">{emo_score:.0f}%</span>
            <div style="color:#7C86A8; font-size:13px; margin-top:8px;">You maintained a neutral expression during the interview.</div>
        </div>
    """, unsafe_allow_html=True)
    detail_placeholders[1].markdown(f"""
        <div class="card">
            <div class="card-title">🎤 Speech Analysis</div>
            <div style="font-weight:600;">{speech_summary_label(data)}</div>
            {progress_bar_html(speech_score, "#4C6FFF")}
            <span style="color:#7C86A8; font-size:12px;">{speech_score:.0f}%</span>
            <div style="color:#7C86A8; font-size:13px; margin-top:8px;">Your speech clarity based on filler word usage.</div>
        </div>
    """, unsafe_allow_html=True)
    confidence = "High" if overall >= 75 else "Medium" if overall >= 50 else "Low"
    detail_placeholders[2].markdown(f"""
        <div class="card">
            <div class="card-title">🛡️ Confidence Analysis</div>
            <div style="font-weight:600;">{confidence}</div>
            {progress_bar_html(overall, "#F5A623")}
            <span style="color:#7C86A8; font-size:12px;">{overall:.0f}%</span>
            <div style="color:#7C86A8; font-size:13px; margin-top:8px;">Combined score from composure and speech quality.</div>
        </div>
    """, unsafe_allow_html=True)
    kw_label = "Good" if keyword_score >= 65 else "Fair" if keyword_score >= 40 else "Limited"
    detail_placeholders[3].markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Keyword Usage</div>
            <div style="font-weight:600;">{kw_label}</div>
            {progress_bar_html(keyword_score, "#2ECC71")}
            <span style="color:#7C86A8; font-size:12px;">{keyword_score:.0f}%</span>
            <div style="color:#7C86A8; font-size:13px; margin-top:8px;">Use of relevant, results-oriented language.</div>
        </div>
    """, unsafe_allow_html=True)


def render_feedback(points):
    html = '<div class="card"><div class="card-title">💬 Feedback</div>'
    for kind, text in points:
        icon = "✅" if kind == "good" else "🟠"
        cls = "fb-good" if kind == "good" else "fb-warn"
        html += f'<div class="fb-item"><span class="{cls}">{icon}</span> {text}</div>'
    html += '</div>'
    feedback_placeholder.markdown(html, unsafe_allow_html=True)


def render_final_score(score_10, report_json=None):
    label = "Excellent" if score_10 >= 8 else "Good" if score_10 >= 6 else "Needs Work"
    html = f"""
        <div class="card" style="text-align:center;">
            <div class="card-title" style="justify-content:center;">🏆 Final Score</div>
            <div style="font-size:44px; font-weight:700; color:white;">{score_10}<span style="font-size:16px; color:#7C86A8;">/10</span></div>
            <div style="color:#2ECC71; font-weight:600; margin-bottom:14px;">{label}</div>
        </div>
    """
    final_placeholder.markdown(html, unsafe_allow_html=True)


# ============================================================
# MAIN LOOP
# ============================================================
if st.session_state.running:
    data["session_active"] = True
    badge.markdown('<span class="badge-active"><span class="badge-dot"></span>Active</span>', unsafe_allow_html=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open camera.")
    else:
        frame_count = 0
        fps_timer = time.time()
        fps_counter = 0
        fps_display = 0
        current_emotion = "-"
        history_scores = load_score_history()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break

            frame_count += 1
            fps_counter += 1
            if time.time() - fps_timer >= 1:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer = time.time()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (46, 204, 113), 2)
                if frame_count % EMOTION_ANALYSIS_EVERY_N_FRAMES == 0:
                    face_roi = frame[y:y + h, x:x + w]
                    if face_roi.size != 0:
                        try:
                            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                            current_emotion = result[0]['dominant_emotion']
                            data["emotion_log"].append(current_emotion)
                        except Exception:
                            pass
                cv2.putText(frame, "Face Detected", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 204, 113), 2)

            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            elapsed = int(time.time() - data["start_time"])
            meta_placeholder.markdown(
                f'<span style="color:#2ECC71;">● Analyzing... {elapsed//60:02d}:{elapsed%60:02d}</span>'
                f'<span style="float:right; color:#7C86A8;">FPS: {fps_display}</span>',
                unsafe_allow_html=True
            )

            overall, emo_score, speech_score = compute_overall(data)
            keyword_score = compute_keyword_score(data)
            data["score_trail"].append(overall)
            if len(data["score_trail"]) > 30:
                data["score_trail"] = data["score_trail"][-30:]

            score_10 = round(overall / 10, 1)
            tag = "Good Performance" if overall >= 60 else "Needs Improvement"
            tag_color = "#2ECC71" if overall >= 60 else "#F5A623"

            gauge_placeholder.plotly_chart(gauge_chart(score_10), use_container_width=True, key=f"gauge_{frame_count}")
            gauge_text_placeholder.markdown(
                f'<div class="gauge-score">{score_10}</div><div class="gauge-outof">/10</div>'
                f'<div class="gauge-tag" style="color:{tag_color};">{tag}</div>',
                unsafe_allow_html=True
            )

            render_side_cards(overall, emo_score, speech_score, current_emotion.capitalize())
            render_detailed_cards(emo_score, speech_score, overall, keyword_score)
            render_feedback(build_feedback(data, emo_score, speech_score))
            history_placeholder.markdown('<div class="card"><div class="card-title">📈 Score History</div>', unsafe_allow_html=True)
            history_placeholder.plotly_chart(history_chart(history_scores, score_10), use_container_width=True, key=f"hist_{frame_count}")
            render_final_score(score_10)

            time.sleep(0.03)
            st.session_state.running = st.session_state.get("running", False)

        cap.release()
else:
    data["session_active"] = False
    badge.markdown('<span class="badge-active" style="background:rgba(124,134,168,.12); color:#7C86A8;">'
                    '<span class="badge-dot" style="background:#7C86A8;"></span>Idle</span>', unsafe_allow_html=True)
    frame_placeholder.info("Toggle 'Start / Stop Analysis' above to begin your session.")
    meta_placeholder.empty()

    overall, emo_score, speech_score = compute_overall(data)
    keyword_score = compute_keyword_score(data)
    score_10 = round(overall / 10, 1)
    tag = "Good Performance" if overall >= 60 else "Needs Improvement"
    tag_color = "#2ECC71" if overall >= 60 else "#F5A623"

    gauge_placeholder.plotly_chart(gauge_chart(score_10), use_container_width=True, key="gauge_idle")
    gauge_text_placeholder.markdown(
        f'<div class="gauge-score">{score_10}</div><div class="gauge-outof">/10</div>'
        f'<div class="gauge-tag" style="color:{tag_color};">{tag}</div>',
        unsafe_allow_html=True
    )
    render_side_cards(overall, emo_score, speech_score, "Neutral")
    render_detailed_cards(emo_score, speech_score, overall, keyword_score)
    render_feedback(build_feedback(data, emo_score, speech_score))

    history_scores = load_score_history()
    history_placeholder.markdown('<div class="card"><div class="card-title">📈 Score History</div>', unsafe_allow_html=True)
    history_placeholder.plotly_chart(history_chart(history_scores, score_10), use_container_width=True, key="hist_idle")
    render_final_score(score_10)

    if st.button("💾 Save Report", use_container_width=True):
        report = save_report(data)
        st.success(f"Report saved to {report['saved_to']}")
        st.json(report)