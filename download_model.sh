#!/usr/bin/env bash
# Attempt to download a recommended YOLOv8 weight from Ultralytics (user must have internet)
# Modify URL below if you have a custom model link.
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
mkdir -p models
echo "Downloading model from $MODEL_URL ..."
curl -L -o models/best.pt $MODEL_URL
echo "Saved models/best.pt. Replace with your real 'best.pt' for production."