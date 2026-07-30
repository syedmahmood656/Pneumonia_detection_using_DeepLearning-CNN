Pneumonia detection using CNN architecture - DeepLearning

---
title: ExplainX-CXR
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.7.1
app_file: frontend/app.py
pinned: false
license: apache-2.0
---

# ExplainX-CXR — Explainable Chest X-Ray Classifier

Custom CNN trained from scratch on NIH ChestX-ray14 subset.

# 🫁 ExplainX-CXR: Explainable Chest X-Ray Pathology Classifier

> Production-ready, end-to-end multi-label chest X-ray abnormality classifier built from scratch using custom deep learning architectures, ONNX INT8 quantization, and real-time Grad-CAM explainability.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Architecture & Pipeline](#-architecture--pipeline)
- [Tech Stack](#-tech-stack)
- [Key Features](#-key-features)
- [Dataset & Data Engineering](#-dataset--data-engineering)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Installation & Quickstart](#-installation--quickstart)
- [API Reference](#-api-reference)
- [Disclaimer](#-disclaimer)

---

## 🩺 Project Overview

**ExplainX-CXR** addresses the challenge of automated multi-label screening for chest pathologies. The primary goal is to provide fast, highly accurate probability scores across 14 distinct lung/heart conditions while generating visual heatmap overlays to build clinical trust.

### 🎯 Key Design Choices:
* **Zero Pretrained Weights:** The core convolutional network (`CXRNet`) is engineered and trained completely from scratch without transferring weights from ImageNet.
* **Multi-Label Binary Classification:** Evaluates all 14 condition classes independently rather than forcing a single diagnosis per image.
* **Quantized Edge Deployment:** Compressed via INT8 post-training quantization to reduce footprint to **~15 MB** while enabling **<30ms CPU latency** per scan.
* **Visual Explainability:** Integrates **Grad-CAM (Gradient-Weighted Class Activation Mapping)** to highlight anatomically relevant regions driving model predictions.

---

## 🏗 Architecture & Pipeline

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA ENGINEERING PIPELINE                       │
│                                                                        │
│   ┌──────────────┐      ┌─────────────────┐      ┌─────────────────┐   │
│   │ Raw NIH      │ ───> │ Stratified      │ ───> │ CLAHE Contrast  │   │
│   │ ChestX-ray14 │      │ Sampling Subset │      │ Enhancement     │   │
│   └──────────────┘      └─────────────────┘      └─────────────────┘   │
│                                                           │            │
│                                                           ▼            │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │        HDF5 Compressed Single-File Container (.h5)             │   │
│   └────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Stream Data (Lazy Loading)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  MODELLING RUNTIME (CXRNet Custom CNN)                 │
│                                                                        │
│   ┌────────────────────┐      ┌───────────────────┐                    │
│   │ Input Tensor       │ ───> │ Conv Stage 1 & 2  │ ───> ...           │
│   │ (1, 1, 224, 224)   │      │ (32 → 64 Channels)│                    │
│   └────────────────────┘      └───────────────────┘                    │
│                                         │                              │
│                                         ▼                              │
│                               ┌───────────────────┐                    │
│                         ... ──│ Conv Stage 3 & 4  │ ───> ...           │
│                               │(128→256 Channels) │                    │
│                               └───────────────────┘                    │
│                                         │                              │
│                                         ▼                              │
│                               ┌───────────────────┐                    │
│                         ... ──│   Conv Stage 5    │ ───┐               │
│                               │  (512 Channels)   │    │               │
│                               └───────────────────┘    │               │
│                                         │              │ Forward       │
│                                         │ Hooks        │ Pass          │
│                                         ▼              ▼               │
│   ┌────────────────────┐      ┌───────────────────┐  ┌───────────────┐ │
│   │ Clinical Display   │ <─── │   Grad-CAM Tool   │  │Global Avg Pool│ │
│   │ Heatmap Overlay    │      │ (Attribution Map) │  │  & Flatten    │ │
│   └────────────────────┘      └───────────────────┘  └───────┬───────┘ │
│                                                              │         │
│                                                              ▼         │
│   ┌────────────────────┐      ┌───────────────────┐  ┌───────────────┐ │
│   │ Sigmoid Prob Matrix│ <─── │Per-Class Threshold│ <│Fully Connected│ │
│   │ (14 Target Labels) │      │   Val-Tuning      │  │ Classifier    │ │
│   └────────────────────┘      └───────────────────┘  └───────────────┘ │
└────────────────────────────────────────────────────────────────────────┘


# 🛠 Tech Stack
Framework: PyTorch & PyTorch Lightning

Data Processing: OpenCV, Albumentations, h5py, pandas, scikit-learn

Loss & Metrics: Custom Focal Loss, torchmetrics (Macro AUC-ROC, F1-Score)

Optimization & Serialization: ONNX Runtime, Dynamic INT8 Quantization

Backend API: FastAPI & Uvicorn

Frontend UI: Gradio

Experiment Tracking: Weights & Biases (W&B)

# 📊 Dataset & Data Engineering

The pipeline uses a preprocessed subset of the **NIH ChestX-ray14** dataset.

**Stratified Sampling**: Retains 100% of rare pathologies (Hernia, Pneumonia, Fibrosis), while capping majority "No Finding" instances at 2,000 to resolve severe class imbalance.

**CLAHE Enhancement:** Applies Contrast Limited Adaptive Histogram Equalization locally to emphasize subtle anatomical contrast variations in dense lung regions.

**High-Speed HDF5 Storage:** Packs processed $224 \times 224$ images into a single .h5 binary database container (cxr_dataset.h5), speeding up training I/O streaming by ~40%.

**Data Augmentation:** Applies anatomically plausible transforms via albumentations (horizontal flips, small rotations, contrast shifts, spatial distortions).

**🚀 Installation & Quickstart**
**1. Clone the Repository**
Bash
git clone [https://github.com/your-username/CNN-Pneumonia_detector.git](https://github.com/your-username/CNN-Pneumonia_detector.git)
cd CNN-Pneumonia_detector

**2 Set Up Virtual Environment & Dependencies**

python -m venv venv
On Windows:
.\venv\Scripts\activate
On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

**3. Run Data Pipeline & Preprocessing**
Bash
Generate stratified splits
python -c "from src.data.sampling import create_stratified_subset; create_stratified_subset('data/raw/Data_Entry_2017.csv')"

Compile HDF5 Binary Storage
python scripts/build_hdf5.py

**4. Launch the Backend API**
Bash
uvicorn api.main:app --reload --port 8000

**5. Launch the Web Interface**
In a separate terminal:

Bash
python frontend/app.py

**📡 API Reference**
POST /predict
Upload a grayscale chest X-ray image file (.png, .jpg, .jpeg) to receive diagnostic prediction scores.

Example Request (Python):
Python
import requests

url = "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)"
files = {"file": open("sample_xray.png", "rb")}
response = requests.post(url, files=files)
print(response.json())

**Example Response Body:**

{
  "predictions": {
    "Cardiomegaly": {
      "probability": 0.8124,
      "threshold": 0.4700,
      "is_positive": true
    },
    "Effusion": {
      "probability": 0.6210,
      "threshold": 0.4500,
      "is_positive": true
    },
    "Pneumonia": {
      "probability": 0.0821,
      "threshold": 0.3200,
      "is_positive": false
    }
  },
  "detected_findings": ["Cardiomegaly", "Effusion"],
  "model_version": "1.0.0-INT8"
}


**⚠️ Disclaimer**
Research & Portfolio Demo Only: This software is intended strictly for academic, research, and technical demonstration purposes. It is not validated, certified, or FDA/CE approved for diagnostic or clinical use. Clinical judgments must always be made by qualified healthcare professionals.