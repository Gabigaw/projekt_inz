import csv, re, sys, time, json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import serial
from serial.serialutil import SerialException

# --- PORT / URZĄDZENIE ---
PORT = "/dev/ttyUSB0"
BAUD = 115200
DEVICE_ID = "gabiplant-1"

# --- ADC / NAPIĘCIA (dopasuj do swojej płytki) ---
ADC_BITS    = 12          # np. ESP32 = 12 bit (0..4095), UNO = 10 bit (0..1023)
ADC_VREF_MV = 3300        # efektywna referencja ADC w mV

# --- KALIBRACJA CZUJNIKA ---
AIR_VALUE   = 3500        # sucho (w powietrzu)
WATER_VALUE = 1200        # mokro (w wodzie)

# --- LOGOWANIE DO CSV ---
SAMPLE_EVERY = 2          # sekundy

# --- ŚCIEŻKI ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = DATA_DIR / "config.json"
RE_CSV = re.compile(r"^measurements_(\d{4})-(\d{2})-(\d{2})\.csv$")

# --- PARSER SUROWEGO WYJŚCIA Z PORTU ---
RE_RAW = re.compile(r"(?:Soil\s*)?RAW\s*=\s*(?P<raw>\d+)", re.I)

def parse_raw(line: str):
    m = RE_RAW.search(line)
    return int(m.group("raw")) if m else None

def raw_to_millivolts(raw: int) -> int:
    max_count = (1 << ADC_BITS) - 1
    if raw < 0: raw = 0
    if raw > max_count: raw = max_count
    return int(round((raw / max_count) * ADC_VREF_MV))

def clamp_to_calibration(raw: int) -> int:
    lo, hi = sorted((WATER_VALUE, AIR_VALUE))
    return max(lo, min(hi, raw))

def raw_to_pct_by_calibration(raw: int) -> float:
    """
    100% = mokro (WATER_VALUE), 0% = sucho (AIR_VALUE)
    Używane TYLKO do decyzji i JSON-a, NIE trafia do CSV.
    """
    lo, hi = sorted((WATER_VALUE, AIR_VALUE))   # lo = mokro, hi = sucho
    r = clamp_to_calibration(raw)
    return 100.0 * (hi - r) / (hi - lo) if hi != lo else 0.0

def ensure_daily_csv(now_utc: datetime) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / now_utc.strftime("measurements_%Y-%m-%d.csv")
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp_utc","device_id","raw","millivolts"])
    return path

def write_latest_json(now_utc: datetime, device_id: str, raw: int, mv: int, pct: float, status: str):
    obj = {
        "timestamp_utc": now_utc.isoformat(timespec="seconds"),
        "device_id": device_id,
        "raw": int(raw),
        "millivolts": int(mv),
        "pct": round(pct, 1),
        "status": status,
    }
    with (DATA_DIR / "latest.json").open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

# --- KONFIG (progi + histereza + telegram) ---
def load_config_file() -> dict:
    cfg = {"thresh_on": 70.0, "hysteresis": 5.0,
           "telegram": {"enabled": False, "bot_token": "", "chat_id": "", "repeat_min": 360}}
    try:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                raw = json.load(f)

            # progi
            t = float(raw.get("thresh_on", cfg["thresh_on"]))
            h = float(raw.get("hysteresis", cfg["hysteresis"]))
            cfg["thresh_on"] = max(0.0, min(100.0, t))
            cfg["hysteresis"] = max(0.0, min(50.0, h))

            # telegram (opcjonalny)
            tg = raw.get("telegram", {})
            cfg["telegram"]["enabled"]   = bool(tg.get("enabled", cfg["telegram"]["enabled"]))
            cfg["telegram"]["bot_token"] = str(tg.get("bot_token", "")).strip()
            cfg["telegram"]["chat_id"]   = str(tg.get("chat_id", "")).strip()
            try:
                cfg["telegram"]["repeat_min"] = int(tg.get("repeat_min", cfg["telegram"]["repeat_min"]))
            except Exception:
                pass
    except Exception as e:
        print(f"[!] Błąd czytania {CONFIG_PATH.name}: {e} (używam domyślnych)")
    return cfg

# --- RETENCJA CSV ---
def cleanup_old_csvs(today_utc_date, retention_days: int = 7):
    cutoff = today_utc_date - timedelta(days=retention_days)
    kept, deleted = 0, 0
    for p in DATA_DIR.glob("measurements_*.csv"):
        m = RE_CSV.match(p.name)
        if not m:
            continue
        y, mo, d = map(int, m.groups())
        try:
            file_date = datetime(y, mo, d).date()
        except ValueError:
            continue
        if file_date < cutoff:
            try:
                p.unlink()
                deleted += 1
                print(f"[i] Usunięto stary plik: {p.name}")
            except Exception as e:
                print(f"[!] Nie udało się usunąć {p.name}: {e}")
        else:
            kept += 1
    if deleted:
        print(f"[i] Retencja: zachowano {kept} plików, usunięto {deleted}.")

# --- Wysyłka Telegram ---
def send_telegram(text: str, token: str, chat_id: str) -> bool:
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        try:
            import requests
            r = requests.post(url, json=payload, timeout=10)
            if not r.ok:
                print(f"[!] Telegram HTTP {r.status_code}: {r.text[:200]}")
            return r.ok
        except Exception:
            # fallback bez requests
            import json as _json, urllib.request
            req = urllib.request.Request(
                url,
                data=_json.dumps(payload).encode("utf-8"),
                headers={"Content-Type":"application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return 200 <= resp.status < 300
    except Exception as e:
        print(f"[!] Telegram błąd: {e}")
        return False

def open_serial_blocking():
    while True:
        try:
            print(f"[i] Czekam na port {PORT} @ {BAUD}…")
            ser = serial.Serial(PORT, BAUD, timeout=2, exclusive=True)
            try:
                ser.setDTR(False); ser.setRTS(False)
            except Exception:
                pass
            print("[i] Połączono.")
            return ser
        except SerialException as e:
            print(f"[!] {e} – ponawiam za 2s")
            time.sleep(2)

def main():
    last_write = 0.0
    alarm = False
    last_cleanup_day = None

    # stan powiadomień Telegram
    prev_alarm = False
    last_alert_ts = 0.0

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # cache konfiguracji + mtime
    cfg = load_config_file()
    cfg_mtime = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else 0.0
    print(f"[i] Start z configiem: {cfg}")

    while True:
        try:
            with open_serial_blocking() as ser:
                while True:
                    line = ser.readline().decode(errors="ignore").strip()
                    if not line:
                        continue
                    raw = parse_raw(line)
                    if raw is None:
                        continue

                    now_utc = datetime.now(timezone.utc)

                    # hot-reload configu
                    try:
                        mtime = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else 0.0
                        if mtime != cfg_mtime:
                            cfg = load_config_file()
                            cfg_mtime = mtime
                            print(f"[i] Załadowano config: {cfg}")
                    except Exception:
                        pass

                    if time.time() - last_write >= SAMPLE_EVERY:
                        mv  = raw_to_millivolts(raw)
                        pct = raw_to_pct_by_calibration(raw)

                        # progi z configu
                        THRESH_ON  = cfg["thresh_on"]
                        THRESH_OFF = min(100.0, THRESH_ON + cfg["hysteresis"])

                        # histereza (stan alarmu)
                        if not alarm and pct < THRESH_ON:  alarm = True
                        if  alarm and pct > THRESH_OFF:    alarm = False
                        status = "PODLEJ" if alarm else "OK"

                        # --- TELEGRAM ---
                        tg = cfg.get("telegram", {})
                        enabled = bool(tg.get("enabled", False))
                        token   = tg.get("bot_token", "")
                        chat_id = tg.get("chat_id", "")
                        repeat_min = int(tg.get("repeat_min", 360))
                        now_ts = time.time()

                        if enabled and token and chat_id:
                            if alarm and not prev_alarm:
                                msg = (f"🌱 GabiPlant: PODLEJ\n"
                                       f"Wilgotność: {pct:.1f}%  (próg {THRESH_ON:.1f}%)")
                                if send_telegram(msg, token, chat_id):
                                    last_alert_ts = now_ts
                            elif alarm and (now_ts - last_alert_ts) >= repeat_min * 60:
                                msg = (f"🌱 Przypomnienie: nadal PODLEJ\n"
                                       f"Aktualnie {pct:.1f}%  (próg {THRESH_ON:.1f}%)")
                                if send_telegram(msg, token, chat_id):
                                    last_alert_ts = now_ts
                            elif (not alarm) and prev_alarm:
                                send_telegram(
                                    f"🌱 OK: wilgotność wróciła do {pct:.1f}% (OFF {THRESH_OFF:.1f}%)",
                                    token, chat_id
                                )
                        prev_alarm = alarm

                        # CSV
                        csv_path = ensure_daily_csv(now_utc)
                        with csv_path.open("a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([
                                now_utc.isoformat(timespec="seconds"),
                                DEVICE_ID,
                                raw,
                                mv
                            ])

                        # JSON
                        write_latest_json(now_utc, DEVICE_ID, raw, mv, pct, status)

                        # Retencja raz dziennie
                        today = now_utc.date()
                        if last_cleanup_day != today:
                            cleanup_old_csvs(today, retention_days=7)
                            last_cleanup_day = today

                        last_write = time.time()
                        print(f"{now_utc.isoformat(timespec='seconds')}  raw={raw}  mv={mv}  pct={pct:.1f}  "
                              f"status={status} (on={THRESH_ON:.1f} off={THRESH_OFF:.1f})  -> {csv_path}")

        except SerialException as e:
            print(f"[!] Utrata połączenia: {e} – łączę ponownie…")
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n[i] Stop (Ctrl+C).")
            sys.exit(0)

if __name__ == "__main__":
    main()
