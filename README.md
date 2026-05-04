# AI Microclimate Anomaly Detection

An AI-powered Smart Cities project for detecting urban microclimate anomalies such as smog, haze, smoke, and related air-quality irregularities using computer vision, sensor data, and a simple interactive dashboard.

## Overview

This project was built as part of a Smart Cities software project. The goal is to monitor urban environmental conditions by combining:

- image-based anomaly detection
- air-quality and weather-related sensor data
- fusion logic for risk estimation
- dashboard-based visualization and feedback

The system is designed to identify possible microclimate anomalies from visual input and environmental readings, then present the results in an interpretable way.

## Problem Statement

Urban areas frequently experience local environmental issues such as smog, haze, smoke, and pollution spikes. Traditional monitoring systems often rely only on sparse sensor stations or isolated dashboards, which can make detection slow, less explainable, and less adaptive.

This project addresses that gap by building a software-based monitoring workflow that combines visual detection with sensor data for better anomaly assessment.

## Solution

The implemented solution includes:

- YOLO-based visual anomaly detection for identifying smoke/smog-like patterns in images
- preprocessing and analysis of air-quality sensor data
- fusion logic to combine visual confidence with environmental readings
- a dashboard interface for displaying detections, metrics, and outputs
- supporting notebooks for experimentation, preprocessing, and fusion design

## Features

- Detects urban anomaly patterns from input images
- Uses YOLOv8-based workflow for model training and inference
- Processes and scales sensor/environmental data
- Combines multimodal signals into a risk/anomaly score
- Includes notebooks documenting the project pipeline
- Supports feedback logging and graph generation

## Tech Stack

- Python
- YOLOv8 / Ultralytics
- Roboflow
- Pandas
- NumPy
- Scikit-learn
- OpenCV
- Streamlit
- Matplotlib
- Jupyter Notebook

## Project Structure

```bash
MICROCLIMATE_ANOMALY_DETECTION/
├── data/
│   ├── clean/
│   └── raw/
├── images/
├── models/
│   ├── fire-and-smoke-detection-yolov8/
│   └── Smog-detection-4/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_fusion_design.ipynb
├── .gitignore
├── app_final.py
├── download_model_1.py
├── download_model_2.py
├── feedback.csv
├── graphs.py
├── requirements.txt
└── train.py
```

## Model and Dataset Sources

This repository does **not** include the full datasets or model weight files in order to keep the repo lightweight.

### 1. Fire and Smoke Detection Model
The fire and smoke detection model used in this project was **directly downloaded** and integrated into the workflow for detection/inference.

- Source / credit: **https://universe.roboflow.com/senior-design-project-cavtw/fire-and-smoke-detection-yolov8**
- Model / project name: **Fire and Smoke detection-yolov8 Computer Vision Dataset**
- Usage: Used directly for fire/smoke-related visual detection

### 2. Smog Detection Dataset
The smog detection component was built differently.

- The **smog detection dataset** was first downloaded in YOLOv8 format.
- That dataset was then used for **custom training** using `train.py`.
- The resulting trained weights (for example `best.pt`) were generated after training and used for inference.

- Dataset source / credit: **https://universe.roboflow.com/final-year-project-4tbnx/smog-detection**
- Dataset / project name: **Smog detection Computer Vision Model**
- Training script: `train.py`

### Download Scripts
The repository contains helper scripts for obtaining external resources:

- `download_model_1.py` → used for downloading the required fire/smoke model or related asset
- `download_model_2.py` → used for downloading the smog detection dataset in YOLO format

## Notes on Excluded Files

To keep the repository lightweight, the following are **not included** in GitHub:

- datasets
- images
- trained model weights (`.pt` files)
- large generated outputs
- local cache files

If you want to run the full project, you will need to download the required model/dataset files separately using the provided scripts or source links.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ai-microclimate-anomaly-detection.git
cd ai-microclimate-anomaly-detection
```

Create and activate a virtual environment:

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

Typical dependencies used in this project include:

```txt
pandas
numpy
scikit-learn
joblib
opencv-python
matplotlib
streamlit
ultralytics
roboflow
python-dotenv
```

## How to Run

### Run the main application
```bash
python app_final.py
```

or, if your app is Streamlit-based:

```bash
streamlit run app_final.py
```

### Run graph generation
```bash
python graphs.py
```

### Open notebooks
```bash
jupyter notebook
```

## AI in the Project

AI is central to this project in two ways:

### 1. In the solution
- YOLO-based object detection is used for visual anomaly recognition.
- machine learning preprocessing is used for scaling and preparing sensor data.
- multimodal fusion combines image and sensor information for anomaly estimation.

### 2. In the development workflow
- AI-assisted ideation and coding support were used to structure modules, refine logic, and accelerate prototyping.
- the project was developed iteratively through experimentation in notebooks and code modules.

## Repository Purpose

This repository is intended to showcase:

- the codebase
- the project workflow
- the notebooks used for experimentation
- the model/dataset integration logic
- the overall architecture of the system

It is not intended to serve as a full dataset/model hosting repository.

## Future Improvements

- improve detection accuracy with more robust custom training
- add stronger fusion strategies between image and sensor inputs
- improve dashboard usability and deployment
- add evaluation metrics and testing pipelines
