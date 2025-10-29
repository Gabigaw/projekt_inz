import csv, os, re, sys, time, json
from datetime import datetime, timezone
import serial
from serial.serialutil import SerialException

# --- USTAWIENIA ---
PORT = "COM3"          # <--- zmień jeśli trzeba
BAUD = 115200
DEVICE_ID = "gabiplant-1"

DATA_DIR = "data"
SAMPLE_EVERY = 2      # sekundy (co ile zapisywać)
RAW_DRY = 3500         # robocza kalibracja (zmienisz po zakupie rośliny)
RAW_WET = 1200

THRESH_ON  = 30.0      # % – poniżej PODLEJ
THRESH_OFF = 35.0      # % – powyżej OK

RE_FULL = re.compile(r"RAW\s*=\s*(?P<raw>\d+).*?Moisture\s*=\s*(?P<pct>[\d\.]+)", re.I)
RE_RAW  = re.compile(r"(Soil\s*)?RAW\s*=\s*(?P<raw>\d+)", re.I)

def map_raw_to_pct(raw:int)->float:
    lo, hi = sorted((RAW_WET, RAW_DRY))
    raw = max(lo, min(hi, raw))
    return 100.0 * (hi - raw) / (hi - lo)

def parse_line(line:str):
    m = RE_FULL.search(line)
    if m:
        return int(m.group("raw")), float(m.group("pct"))
    m = RE_RAW.search(line)
    if m:
        return int(m.group("raw")), None
    return None, None

def ensure_daily_csv(now_utc:datetime):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, now_utc.strftime("measurements_%Y-%m-%d.csv"))
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp_utc","device_id","raw","pct","status"])
    return path

def write_latest_json(now_utc, raw, pct, status):
    obj = {
        "timestamp_utc": now_utc.isoformat(timespec="seconds"),
        "device_id": DEVICE_ID,
        "raw": raw,
        "pct": round(pct,1),
        "status": status
    }
    with open(os.path.join(DATA_DIR, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def open_serial_blocking():
    while True:
        try:
            print(f"[i] Czekam na port {PORT} @ {BAUD}…")
            ser = serial.Serial(PORT, BAUD, timeout=2)
            print("[i] Połączono.")
            return ser
        except SerialException as e:
            print(f"[!] {e} – ponawiam za 2s")
            time.sleep(2)

def main():
    alarm = False
    last_write_t = 0.0

    while True:  # auto-reconnect
        try:
            with open_serial_blocking() as ser:
                while True:
                    line = ser.readline().decode(errors="ignore").strip()
                    if not line:
                        continue
                    raw, pct = parse_line(line)
                    if raw is None:
                        continue

                    if pct is None:
                        pct = map_raw_to_pct(raw)

                    # histereza statusu
                    if not alarm and pct < THRESH_ON:  alarm = True
                    if  alarm and pct > THRESH_OFF:    alarm = False
                    status = "PODLEJ" if alarm else "OK"

                    now_utc = datetime.now(timezone.utc)
                    # zapis co SAMPLE_EVERY sekund
                    if time.time() - last_write_t >= SAMPLE_EVERY:
                        csv_path = ensure_daily_csv(now_utc)
                        with open(csv_path, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([
                                now_utc.isoformat(timespec="seconds"),
                                DEVICE_ID, raw, round(pct,1), status
                            ])
                        write_latest_json(now_utc, raw, pct, status)
                        last_write_t = time.time()
                        print(f"{now_utc.isoformat(timespec='seconds')}  raw={raw}  pct={pct:.1f}  status={status}")

        except SerialException as e:
            print(f"[!] Utrata połączenia: {e} – łączę ponownie…")
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n[i] Stop (Ctrl+C).")
            sys.exit(0)

if __name__ == "__main__":
    main()
