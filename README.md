# WebGazeGuard
### A Webcam-Based System for Real-Time Eye Strain Risk Detection via Multimodal Behavioral Signals

## Overview

This project presents a **multimodal eye fatigue detection system** that combines:

* **Computer Vision (CV)** signals such as blink rate, head pose, and viewing distance
* **Deep learning gaze estimation (CNN)**
* **Natural Language Processing (NLP)** using **PhoBERT** for user-reported symptoms

The system integrates these modalities to estimate an overall **eye strain risk level** and generate **user-friendly recommendations**.

This repository is designed to support:

* **Research reproducibility and experimental evaluation** (training notebooks, evaluation, analysis)
* **Real-time inference and demo** (serving pipeline, web integration)

---

## Project Structure

```
project/
├── vision/        # Classical CV feature extraction (blink, head pose, distance)
├── temporal/      # Temporal aggregation and filtering of CV signals
├── ml_cnn/        # CNN gaze estimation (training, inference, evaluation)
├── nlp/           # PhoBERT preprocessing, classifier, and report generation
├── fusion/        # Risk fusion and coaching recommendation logic
├── web/           # Backend API, frontend UI, and integration layer
├── runners/       # Offline video and real-time webcam pipelines
├── core/          # Shared schemas, configs, and main processing pipeline
├── analysis/      # Statistics, plots, realtime metrics, TF-IDF baseline
├── notebooks/     # Training and experimental notebooks (research only)
├── samples/       # Demo scripts
└── main.py        # Entry point for system execution
```

---

## System Pipeline

### 1. Computer Vision

Video frames are processed to extract:

* Eye landmarks
* Blink detection and blink rate
* Head pose and viewing distance

Temporal smoothing is applied before producing CV-based fatigue indicators.

### 2. Deep Learning Gaze Estimation

A CNN-based gaze model predicts **visual attention direction** and contributes to fatigue assessment.

### 3. NLP Symptom Analysis

User-provided text is processed through:

```
text preprocessing → PhoBERT encoder → severity classifier → report generation
```

The classifier outputs:

* **Severity level** (low / medium / high)
* **The classifier predicts eye strain severity levels (low / medium / high)
based on user symptom descriptions.

### 4. Multimodal Risk Fusion

All signals are combined into a **single fatigue risk score**, followed by **personalized coaching recommendations**.

---

## Training vs Inference

### Training (Research Only)

Located in:

```
notebooks/
ml_cnn/train.py
analysis/
```

Used for:

* Model training
* Evaluation metrics
* Visualization and ablation studies

**Not executed during demo/runtime.**

---

### Inference (Runtime System)

Runtime execution uses:

```
nlp/classifier.py
fusion/
core/pipeline.py
web/
```

Process:

```
Load trained checkpoints → Receive input → Predict → Fuse risk → Return results
```

No retraining is required for demo execution.

---

## Installation

```bash
git clone <repo-url>
cd project
pip install -r requirements.txt
```

---

## Running the Web Demo

The web demo consists of a **FastAPI backend** and a **Vite frontend**.

Two terminals are required:
- Terminal 1 → backend
- Terminal 2 → frontend

---

### 1. Start the Backend

Open **Git Bash in the project root** and activate the virtual environment:

```bash
source venv/Scripts/activate
```

Start the FastAPI backend:

```bash
python -m uvicorn web.back_end.app.main:app --reload
```

If the backend starts successfully, open the API documentation in your browser:

```
http://127.0.0.1:8000/docs
```

---

### 2. Obtain an Access Token

In another terminal, activate the environment again:

```bash
source venv/Scripts/activate
```

Register a test user using `curl`:

```bash
curl -X POST http://127.0.0.1:8000/api/auth/register \
-H "Content-Type: application/json" \
-d "{\"username\":\"test\",\"email\":\"test@example.com\",\"password\":\"test123\"}"
```

The response will return a JSON object containing the **access token**.

Example response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

Copy this token for use in the web interface.

---

### 3. Start the Frontend

Open **another Git Bash terminal** and navigate to the frontend directory:

```bash
cd web/front_end
```

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

The frontend will start at:

```
http://localhost:5173
```

---

### 4. Run the Demo

1. Open the frontend in your browser.
2. Paste the **access token** obtained earlier.
3. Allow webcam access.
4. Start the webcam stream.

The system will perform **real-time eye strain risk estimation** using multimodal behavioral signals.

## Output

The system returns structured JSON including:

* Fatigue **severity label**
* **Risk score**
* **The classifier predicts eye strain severity levels (low / medium / high)
based on user symptom descriptions.
* Optional CV and NLP diagnostic metrics

---

## Research Purpose

This repository supports:

* **Multimodal fatigue detection research**
* **Baseline comparison (TF-IDF vs PhoBERT)**
* **Real-time human-computer interaction safety**

---

## License

This project is intended for **academic and research use**.
Please contact the authors for other usage permissions.

---

