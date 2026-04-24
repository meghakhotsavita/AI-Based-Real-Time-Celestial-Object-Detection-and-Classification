from ultralytics import YOLO

# load model
model = YOLO("runs/detect/train/weights/best.pt")

# run detection
results = model.predict(
    source="test.jpg",   # 👈 IMPORTANT CHANGE
    conf=0.3,
    save=True
)

print("Done ✅")
