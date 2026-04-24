from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Image path (VERY IMPORTANT)
image_path = r"dataset/train/images/10_jpg.rf.d9363d5f592cac608afc492f28cc6362.jpg"

# Predict
results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

print("Detection Done ✅")