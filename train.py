from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=60,
    imgsz=416,
    batch=2,
    workers=0,
    device="cpu",
    augment=True,
    mosaic=0
)