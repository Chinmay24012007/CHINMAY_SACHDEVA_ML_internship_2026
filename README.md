# Smart Interview Analyzer

A real-time interview analysis tool that evaluates a candidate's facial emotion and speech quality through a webcam and microphone, producing a live score and a final feedback report.

## Features

- **Real-time face detection** using OpenCV Haar Cascades
- **Facial emotion recognition** (happy, sad, angry, fear, disgust, surprise, neutral) using DeepFace
- **Hybrid speech-to-text**: transcribes live audio using Google's online Speech API, automatically falling back to offline Whisper transcription when there's no internet connection
- **Live composite scoring** combining emotional composure and speech clarity into a single overall score
- **Automated feedback generation** based on session performance
- **Session reports** saved as timestamped JSON files for later review
- **Two interface options**: a Flask web app (studio-monitor style UI with live video streaming) and a Streamlit app (simpler, faster to modify)

## Tech Stack

| Component            | Library           |
|-----------------------|--------------------|
| Face detection         | OpenCV (Haar Cascade) |
| Emotion recognition    | DeepFace |
| Speech-to-text (online)  | SpeechRecognition (Google Web Speech API) |
| Speech-to-text (offline) | OpenAI Whisper |
| Web interface (option 1) | Flask |
| Web interface (option 2) | Streamlit |

## Pre-trained Models Used

This project does not train any models from scratch. It uses pre-trained models, each originally trained on the following datasets:

- **Haar Cascade (face detection)** — trained by OpenCV on a labeled face/non-face image set (Viola-Jones method)
- **DeepFace emotion model** — trained on the FER2013 dataset (~35,000 labeled grayscale facial expression images across 7 emotion classes)
- **Whisper (offline speech-to-text)** — trained by OpenAI on 680,000 hours of multilingual audio with transcripts
- **Google Web Speech API (online speech-to-text)** — proprietary Google model, accessed via API, no local training data involved

## Project Structure

```
smart interview analyzer/
├── camera.py              # Standalone OpenCV version (local window, no web UI)
├── app.py                 # Flask backend (web app version)
├── templates/
│   └── index.html          # Flask frontend page
├── static/
│   ├── style.css            # Flask frontend styling
│   └── app.js               # Flask frontend logic
├── streamlit_app.py       # Streamlit version (single-file web app)
└── reports/                # Auto-generated session reports (JSON), created on first run
```

## Installation

```bash
pip install opencv-python deepface tf-keras SpeechRecognition openai-whisper pyaudio soundfile flask streamlit
```

Whisper also requires `ffmpeg` installed on your system separately (not via pip).

## Usage

### Option 1: Standalone script (camera.py)
```bash
python camera.py
```
Opens a local OpenCV window. Press `q` to quit and print the final report to the terminal.

### Option 2: Flask web app
```bash
python app.py
```
Then open `http://localhost:5000` in a browser. Click "End Session" to view the report card.

### Option 3: Streamlit web app
```bash
streamlit run streamlit_app.py
```
Opens automatically in a browser tab. Check "Start Camera" to begin, uncheck it, then click "Generate Report".

## Scoring Methodology

**Emotional Composure Score (0–100):**
Calculated from the ratio of positive emotions (`neutral`, `happy`) to negative emotions (`fear`, `sad`, `angry`, `disgust`) detected across the session.

**Speech Quality Score (0–100):**
Calculated from filler word density (`um`, `uh`, `like`, `you know`, etc.) relative to total words spoken. Lower filler ratio results in a higher score.

**Overall Score:**
Average of the emotional composure and speech quality scores, also expressed as a Final Score out of 10.

**Confidence Level:**
- High: overall score ≥ 75
- Medium: overall score 50–74
- Low: overall score < 50

**Feedback:**
Auto-generated based on which specific component(s) scored low (composure, filler words, low speech volume, or face visibility).

## Session Reports

Each completed session is saved automatically to `reports/interview_report_<timestamp>.json`, containing:
- Emotion summary
- Speech clarity summary
- Confidence level
- Final score
- Feedback text
- Raw numeric scores

## Limitations

- Emotion detection accuracy depends on lighting, camera angle, and face visibility
- Speech recognition accuracy may vary with background noise or accents
- Filler-word-based speech scoring is a simple heuristic, not a comprehensive measure of communication quality
- No dataset was independently collected or used to validate scoring thresholds — thresholds are manually set and can be tuned

## Possible Future Improvements

- Validate and tune scoring thresholds against a labeled set of real interview recordings
- Add eye contact / gaze tracking
- Add speech pace (words per minute) as a scoring factor
- Export reports as PDF in addition to JSON
- Add historical session comparison/trends dashboard

