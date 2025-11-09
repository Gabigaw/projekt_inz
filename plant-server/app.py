from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import json, datetime

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR.parent / "plant-logger" / "data").resolve()
LATEST = DATA_DIR / "latest.json"
CONFIG = DATA_DIR / "config.json"
SERIES = DATA_DIR / "series_week.json"
FORECAST = DATA_DIR / "forecast.json"

app = FastAPI(title="GabiPlant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# --------- MODELE ----------
class ConfigIn(BaseModel):
    thresh_on: float = Field(30.0, ge=0, le=100)
    hysteresis: float = Field(5.0, ge=0, le=50)

# --------- POMOCNICZE ----------
def _default_config() -> dict:
    return {
        "thresh_on": 30.0,
        "hysteresis": 5.0,
        "telegram": {
            "enabled": False,
            "bot_token": "",
            "chat_id": "",
            "repeat_min": 360
        }
    }

def _clamp_cfg(cfg: dict) -> dict:
    t = float(cfg.get("thresh_on", 30.0))
    h = float(cfg.get("hysteresis", 5.0))
    cfg["thresh_on"] = max(0.0, min(100.0, t))
    cfg["hysteresis"] = max(0.0, min(50.0, h))
    return cfg

def read_config_raw() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG.exists():
        with open(CONFIG, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = _default_config()
    else:
        data = _default_config()
        with open(CONFIG, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def read_config() -> dict:
    # zwraca pełny config, tylko przycina wartości progów
    cfg = read_config_raw()
    return _clamp_cfg(cfg.copy())

def write_config_merge(new_cfg: ConfigIn) -> dict:
    """Aktualizuje TYLKO thresh_on/hysteresis i zapisuje cały plik z zachowaniem innych sekcji."""
    current = read_config_raw()
    current["thresh_on"] = float(new_cfg.thresh_on)
    current["hysteresis"] = float(new_cfg.hysteresis)
    current = _clamp_cfg(current)
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
    return current

# --------- API ----------
@app.get("/api/status")
def get_status():
    if not LATEST.exists():
        raise HTTPException(status_code=404, detail="latest.json not found")
    with open(LATEST, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["server_time_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds")
    data["config"] = read_config()
    return data

@app.get("/api/config")
def get_config():
    return read_config()

@app.get("/api/series/week")
def get_series_week():
    if not SERIES.exists():
        raise HTTPException(404, "series_week.json not found (uruchom analytics.py)")
    return json.loads(SERIES.read_text(encoding="utf-8"))

@app.get("/api/forecast")
def get_forecast():
    if not FORECAST.exists():
        raise HTTPException(404, "forecast.json not found (uruchom analytics.py)")
    return json.loads(FORECAST.read_text(encoding="utf-8"))

@app.post("/api/config")
def post_config(cfg: ConfigIn):
    saved = write_config_merge(cfg)
    # zwróć tylko to, co edytuje UI, plus info że OK
    return {"ok": True, "thresh_on": saved["thresh_on"], "hysteresis": saved["hysteresis"]}

# Statics
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
