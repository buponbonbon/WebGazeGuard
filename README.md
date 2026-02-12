# Multimodal Eye Fatigue Detection System

## Overview

This project presents a **multimodal eye fatigue detection system** that combines:

* **Computer Vision (CV)** signals such as blink rate, head pose, and viewing distance
* **Deep learning gaze estimation (CNN)**
* **Natural Language Processing (NLP)** using **PhoBERT** for user-reported symptoms

The system integrates these modalities to estimate an overall **eye strain risk level** and generate **user-friendly recommendations**.

This repository is designed to support:

* **Research reproducibility** (training notebooks, evaluation, analysis)
* **Real-time inference and demo** (serving pipeline, web integration)
* **Conference submission**

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
* **User-facing bilingual report** (Vietnamese / English)

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

## Running the Demo

### Offline Video

```bash
python runners/offline_video.py
```

### Real-time Webcam

```bash
python runners/online_webcam.py
```

### Full System Entry

```bash
python main.py
```

---

## Output

The system returns structured JSON including:

* Fatigue **severity label**
* **Risk score**
* **Bilingual recommendations** (VI / EN)
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

## Authors

* Computer Vision & Temporal Modeling
* CNN Gaze Estimation
* NLP & Multimodal Fusion
* System Integration & Web Deployment

