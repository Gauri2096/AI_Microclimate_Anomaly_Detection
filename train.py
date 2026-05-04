from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='Smog-detection-4/data.yaml',
    epochs=10,
    imgsz=640,
    val=True,      # Auto-creates validation split
    split='val',   # Splits 20% of train to valid
    plots=True
)
