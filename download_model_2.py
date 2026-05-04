"""
Download Roboflow dataset/model - smog-detection v4 (YOLOv8 format)
"""

import os
from roboflow import Roboflow
from dotenv import load_dotenv
load_dotenv()
model_api_key=os.getenv('MODEL_API_KEY')
def download_roboflow_dataset():
    # Roboflow API Key (get from app.roboflow.com → Workspace → API Keys)
    API_KEY = "model_api_key"  # ← Replace with your key!
    
    # Initialize Roboflow
    rf = Roboflow(api_key=API_KEY)
    
    # Access your specific project
    project = rf.workspace("final-year-project-4tbnx").project("smog-detection")
    version = project.version(4)
    
    # Download YOLOv8 format (creates folder with data.yaml + images/labels)
    print("Downloading smog-detection v4 (YOLOv8)...")
    dataset = version.download("yolov8")
    
    print(f"Downloaded to: {dataset.location}")
    print("Contains: data.yaml, train/, valid/, test/ folders")
    return dataset.location

if __name__ == "__main__":
    # Install if needed: pip install roboflow
    try:
        dataset_path = download_roboflow_dataset()
    except Exception as e:
        print(f"Error: {e}")
        print("Get API key: app.roboflow.com → Workspace → API Keys")
