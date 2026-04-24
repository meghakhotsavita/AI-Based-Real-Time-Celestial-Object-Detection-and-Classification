from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime, os, json, base64, uuid
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI(title="Celestial AI Backend 🚀")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "train",
    "weights",
    "best.pt"
)

# ---------------- LOG SETUP ----------------
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "detections.json")
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

# ---------------- SETTINGS ----------------
SETTINGS_FILE = os.path.join(LOG_DIR, "settings.json")

DEFAULT_SETTINGS = {
    "model_path": MODEL_PATH,
    "confidence_threshold": 0.30,
    "camera_interval": 1500
}

if not os.path.exists(SETTINGS_FILE):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(DEFAULT_SETTINGS, f, indent=2)

def load_settings():
    with open(SETTINGS_FILE) as f:
        return json.load(f)

# ---------------- LOAD MODEL ----------------
model = None

def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print("❌ Model NOT found:", MODEL_PATH)
            model = None
            return

        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully")
        print("📌 Classes:", model.names)

    except Exception as e:
        print("❌ Model loading error:", e)
        model = None

load_model()

# ---------------- MODELS ----------------
class LoginRequest(BaseModel):
    username: str
    password: str

class FrameRequest(BaseModel):
    image: str

# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"message": "Backend Running 🚀"}

# ---------------- STATUS ----------------
@app.get("/api/status")
def status():
    return {
        "backend": "running",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

# ---------------- LOGIN ----------------
@app.post("/api/login")
def login(data: LoginRequest):
    if data.username == "admin" and data.password == "admin":
        return {"success": True}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# ---------------- IMAGE DETECTION ----------------
@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        image = Image.open(file.file).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image")

    detections = generate_detections(image)

    log_detection(file.filename, detections)

    return {
        "success": True,
        "count": len(detections),
        "detections": detections
    }

# ---------------- CAMERA DETECTION ----------------
@app.post("/api/detect_frame")
async def detect_frame(data: FrameRequest):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        header, encoded = data.image.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid frame")

    detections = generate_detections(image)

    log_detection("camera.jpg", detections)

    return {
        "success": True,
        "count": len(detections),
        "detections": detections
    }

# ---------------- DETECTION CORE ----------------
def generate_detections(image):
    results = model.predict(source=image, conf=0.30, device="cpu")

    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            confidence = float(box.conf[0])
            label = model.names[int(box.cls[0])].lower()

            detections.append({
                "label": label,
                "confidence": round(confidence, 2),
                "bbox": [int(x) for x in box.xyxy[0]]
            })

    print("🔍 Detections:", detections)
    return detections

# ---------------- LOG SYSTEM ----------------
def log_detection(filename, detections):
    log = {
        "id": str(uuid.uuid4()),
        "time": datetime.datetime.now().isoformat(),
        "file": filename,
        "detections": detections
    }

    with open(LOG_FILE, "r+") as f:
        data = json.load(f)
        data.append(log)
        f.seek(0)
        json.dump(data, f, indent=2)

# ---------------- LOG APIs ----------------
@app.get("/api/logs")
def logs():
    with open(LOG_FILE) as f:
        return json.load(f)

@app.delete("/api/logs")
def delete_logs():
    with open(LOG_FILE, "w") as f:
        json.dump([], f)
    return {"message": "All logs deleted"}

@app.delete("/api/logs/{log_id}")
def delete_log(log_id: str):
    with open(LOG_FILE, "r+") as f:
        data = json.load(f)

        # remove matching log
        new_data = [log for log in data if log.get("id") != log_id]

        f.seek(0)
        f.truncate()
        json.dump(new_data, f, indent=2)

    return {"message": "Log deleted"}
