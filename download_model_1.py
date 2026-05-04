from roboflow import Roboflow
import os
from dotenv import load_dotenv
load_dotenv()
model_api_key=os.getenv('MODEL_API_KEY')
rf = Roboflow(api_key=model_api_key)
project = rf.workspace("senior-design-project-cavtw").project("fire-and-smoke-detection-yolov8")
version = project.version(1)
dataset = version.download("yolov8")
                