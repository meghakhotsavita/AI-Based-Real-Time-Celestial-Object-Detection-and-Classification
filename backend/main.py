from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    WebSocket,
    WebSocketDisconnect
)
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import json
import random
from datetime import datetime
import os
import uuid

# ================== APP SETUP ==================
app = FastAPI(title="Celestial AI Backend (Simulation Mode)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== LOGIN MODELS ==================
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    username: str
    role: str

# ================== USER DB ==================
USERS_FILE = "users.json"

def load_users():
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

# ================== LOGIN API ==================
@app.post("/login", response_model=LoginResponse)
def login(data: LoginRequest):
    users = load_users()

    for user in users:
        if (
            user["username"] == data.username
            and user["password"] == data.password
        ):
            return {
                "token": str(uuid.uuid4()),
                "username": user["username"],
                "role": user.get("role", "user")
            }

    raise HTTPException(status_code=401, detail="Invalid credentials")

# ================== LOG SETUP ==================
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "detections.json")

os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

# ================== OBJECT DETECTION API ==================
@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    labels = ["Star", "Planet", "Asteroid", "Meteor", "Satellite"]

    detections = []
    for _ in range(random.randint(1, 3)):
        detections.append({
            "label": random.choice(labels),
            "confidence": round(random.uniform(0.7, 0.95), 2),
            "bbox": [
                random.randint(30, 600),   # x
                random.randint(30, 400),   # y
                random.randint(40, 120),   # width
                random.randint(40, 120)    # height
            ]
        })

    log_entry = {
        "time": datetime.now().isoformat(),
        "detections": detections
    }

    try:
        with open(LOG_FILE, "r+") as f:
            data = json.load(f)
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"detections": detections}

# ================== LOGS API ==================
@app.get("/api/logs")
def get_logs():
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================== ROOT ==================
@app.get("/")
def root():
    return {
        "status": "Celestial AI Backend running",
        "mode": "Simulation",
        "logs_file": LOG_FILE
    }

# =========================================================
# 🔥 WEBSOCKET FOR REAL-TIME SKY MAP
# =========================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/sky")
async def sky_socket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# =========================================================
# 🔁 AUTO BROADCAST AFTER EACH DETECTION
# =========================================================
@app.middleware("http")
async def broadcast_after_detection(request, call_next):
    response = await call_next(request)

    if request.url.path == "/api/detect" and request.method == "POST":
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
                if logs:
                    await manager.broadcast(logs[-1])
        except Exception:
            pass

    return response
