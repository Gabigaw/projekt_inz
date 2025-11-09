from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# --- ŚCIEŻKI ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR.parent / "plant-logger" / "data").resolve()
CONFIG   = DATA_DIR / "config.json"
SERIES_OUT   = DATA_DIR / "series_week.json"
FORECAST_OUT = DATA_DIR / "forecast.json"

# --- KALIBRACJA (spójna z logger.py) ---
AIR_VALUE   = 3500  # sucho
WATER_VALUE = 1200  # mokro

def raw_to_pct_by_calibration(raw: int) -> float:
    lo, hi = sorted((WATER_VALUE, AIR_VALUE))   # lo = mokro, hi = sucho
    r = np.clip(raw, lo, hi)
    return 100.0 * (hi - r) / (hi - lo) if hi != lo else 0.0

def read_config() -> dict:
    cfg = {"thresh_on": 70.0, "hysteresis": 5.0}
    if CONFIG.exists():
        try:
            cfg.update(json.loads(CONFIG.read_text(encoding="utf-8")))
        except Exception:
            pass
    # clamp
    cfg["thresh_on"]  = max(0.0, min(100.0, float(cfg.get("thresh_on", 70.0))))
    cfg["hysteresis"] = max(0.0, min(50.0,  float(cfg.get("hysteresis", 5.0))))
    return cfg

def load_csv_last_days(days: int = 9) -> pd.DataFrame:
    """Czyta ostatnie pliki measurements_YYYY-MM-DD.csv (UTC)."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days-1)
    frames = []
    for i in range(days):
        d = start + timedelta(days=i)
        path = DATA_DIR / f"measurements_{d:%Y-%m-%d}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                frames.append(df)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame(columns=["timestamp_utc","device_id","raw","millivolts"])
    df = pd.concat(frames, ignore_index=True)
    # parse time (UTC)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    # wylicz pct (z surowego raw)
    df["pct"] = raw_to_pct_by_calibration(df["raw"].astype(int).values)
    return df.sort_values("timestamp_utc")

def hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Resampling do 1h mediany, wykrycie luk i krótkie wypełnienia."""
    if df.empty:
        idx = pd.date_range(datetime.now(timezone.utc)-timedelta(days=7),
                            periods=7*24+1, freq="1H")
        return pd.DataFrame({"pct":[np.nan]*len(idx)}, index=idx)

    s = df.set_index("timestamp_utc")["pct"].resample("1H").median()
    # krótkie luki do 6h – liniowo
    s_interp = s.interpolate(limit=6, limit_direction="both")
    # zachowaj info o tym, co wypełniliśmy
    filled_mask = s_interp.notna() & s.isna()

    out = pd.DataFrame({"pct": s_interp})
    out["filled"] = False
    out.loc[filled_mask, "filled"] = True
    return out

def detect_waterings(hourly: pd.DataFrame, jump_on: float = 6.0) -> list[str]:
    """Wykrywa podlewanie jako skok > jump_on % w górę (na godzinowej serii)."""
    s = hourly["pct"]
    d = s.diff()
    events = []
    # prosta filtracja: skok w górę i utrzymanie > połowy przez 2h
    for i in range(2, len(d)):
        if d.iat[i] is not np.nan and d.iat[i] >= jump_on:
            # check next 2 hours stability
            window = s.iloc[i:i+3]
            if (window - s.iat[i-1]).median() >= (jump_on * 0.5):
                events.append(s.index[i].isoformat())
    return events

def last_cycle_slice(hourly: pd.DataFrame, waterings_iso: list[str]) -> pd.DataFrame:
    """Zwraca fragment serii od ostatniego podlewania do teraz."""
    if not len(hourly):
        return hourly
    if not waterings_iso:
        # jeśli brak detekcji – użyj ostatnich 3 dni
        start = hourly.index.max() - timedelta(days=3)
        return hourly.loc[hourly.index >= start]
    last_w = pd.Timestamp(waterings_iso[-1])
    return hourly.loc[hourly.index >= last_w]

def fit_linear_slope(cycle: pd.DataFrame) -> float | None:
    """Zwraca nachylenie [%/dzień] (ujemne)."""
    s = cycle["pct"].dropna()
    if len(s) < 6:  # min 6h
        return None
    t0 = s.index[0]
    x = np.array([(t - t0).total_seconds() / 86400.0 for t in s.index])  # dni
    y = s.values
    # polyfit stopnia 1: y = a + b x
    try:
        b, a = np.polyfit(x, y, 1)[0], np.polyfit(x, y, 1)[1]  # ale potrzebny tylko b
    except Exception:
        return None
    return float(b)

def make_forecast(hourly: pd.DataFrame, cfg: dict, slope: float | None) -> dict:
    now = datetime.now(timezone.utc)
    current = float(hourly["pct"].iloc[-1]) if len(hourly) and not np.isnan(hourly["pct"].iloc[-1]) else None
    thresh = float(cfg["thresh_on"])

    eta_days = None
    eta_ts = None
    if (current is not None) and (slope is not None) and slope < 0 and current > thresh:
        eta_days = (thresh - current) / slope  # slope < 0 -> dodatni czas
        if eta_days >= 0 and np.isfinite(eta_days) and eta_days < 60:
            eta_ts = (now + timedelta(days=float(eta_days))).isoformat()
        else:
            eta_days = None

    return {
        "generated_utc": now.isoformat(),
        "current_pct": current,
        "thresh_on": thresh,
        "model": "linear",
        "slope_pct_per_day": slope,
        "eta_days_to_thresh": eta_days,
        "eta_timestamp_utc": eta_ts,
    }

def save_series_week(hourly: pd.DataFrame, waterings: list[str]):
    # utnij do 7 dni wstecz
    end = hourly.index.max() if len(hourly) else pd.Timestamp(datetime.now(timezone.utc))
    start = end - timedelta(days=7)
    h = hourly.loc[(hourly.index >= start) & (hourly.index <= end)].copy()

    # JSON: lista punktów {t,pct,filled}
    records = [{"t": ts.isoformat(), "pct": (None if np.isnan(v) else float(v)), "filled": bool(f)}
               for ts, v, f in zip(h.index, h["pct"].values, h["filled"].values)]

    SERIES_OUT.write_text(json.dumps({
        "points": records,
        "waterings": waterings
    }, ensure_ascii=False), encoding="utf-8")

def main():
    cfg = read_config()
    df = load_csv_last_days(days=9)
    hourly = hourly_series(df)
    waterings = detect_waterings(hourly, jump_on=6.0)
    cycle = last_cycle_slice(hourly, waterings)
    slope = fit_linear_slope(cycle)

    forecast = make_forecast(hourly, cfg, slope)
    FORECAST_OUT.write_text(json.dumps(forecast, ensure_ascii=False), encoding="utf-8")
    save_series_week(hourly, waterings)
    print("[i] analytics: zapisano series_week.json i forecast.json")

if __name__ == "__main__":
    main()
