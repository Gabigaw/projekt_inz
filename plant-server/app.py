from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json, datetime

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR.parent / "plant-logger" / "data").resolve()
LATEST = DATA_DIR / "latest.json"

app = FastAPI(title="GabiPlant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# 1) Najpierw API
@app.get("/api/status")
def get_status():
    if not LATEST.exists():
        raise HTTPException(status_code=404, detail="latest.json not found")
    with open(LATEST, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["server_time_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds")
    return data

# 2) Na końcu mount statyk na "/"
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
