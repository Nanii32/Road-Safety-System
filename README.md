
# 🚦 Two-Wheeler Traffic Rules Violation Detection System

A real-time AI-powered surveillance system designed to detect and report two-wheeler traffic violations such as helmet-less riding, triple riding, and mobile phone usage. The project leverages YOLOv8 for object detection, OCR for license plate recognition, and Streamlit for interactive user visualization.

## 🔍 Features

- Helmet detection
- Triple riding detection
- Mobile usage while driving
- Automated violation reporting interface using Streamlit

## 🧠 Technologies Used

- YOLOv8 (Object Detection)
- Roboflow (Model training & augmentation)
- Python, OpenCV
- Streamlit (Web UI)

## 📁 Dataset

- Motorcycle Rider Dataset (Kaggle, Roboflow)
- Helmet & mobile usage images
- WSN-DS for intrusion classification (for cybersecurity extension)

## ⚙️ Installation

```bash
git clone https://github.com/Nanii32/Road-Safety-System
cd traffic-violation-detector
pip install -r requirements.txt
streamlit run app.py
