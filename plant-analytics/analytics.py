from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
from joblib import load as joblib_load
import numpy as np

# --- ŚCIEŻKI ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR.parent / "plant-logger" / "data").resolve()
CONFIG   = DATA_DIR / "config.json"
SERIES_OUT   = DATA_DIR / "series_week.json"
FORECAST_OUT = DATA_DIR / "forecast.json"
ML_DIR       = (DATA_DIR / "ml").resolve()
ML_CONFIG    = ML_DIR / "ml_config.json"

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

    s = df.set_index("timestamp_utc")["pct"].resample("1h").median()
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

def _compute_hours_since_last_watering(hourly: pd.DataFrame, waterings_iso: list[str]) -> pd.Series:
    """Zwraca serię 'hours_since_last_watering' w godzinach dla każdego punktu."""
    if not waterings_iso:
        # Brak wykrytych podlewań – przyjmij, że "ostatnie podlewanie"
        # było na początku logu (czas od pierwszego pomiaru).
        if not len(hourly):
            return pd.Series([], dtype=float, name="hours_since_last_watering")
        t0 = hourly.index[0]
        vals = [
            (t - t0).total_seconds() / 3600.0
            for t in hourly.index
        ]
        return pd.Series(vals, index=hourly.index, name="hours_since_last_watering")

    waterings_ts = sorted(pd.to_datetime(waterings_iso))
    result = []
    j = 0
    for t in hourly.index:
        # przesuwamy wskaźnik do ostatniego podlewania <= t
        while j + 1 < len(waterings_ts) and waterings_ts[j + 1] <= t:
            j += 1
        if waterings_ts[j] <= t:
            delta_h = (t - waterings_ts[j]).total_seconds() / 3600.0
            result.append(delta_h)
        else:
            result.append(np.nan)
    return pd.Series(result, index=hourly.index, name="hours_since_last_watering")



def _make_feature_vector_for_index(
    hourly: pd.DataFrame,
    cfg: dict,
    hours_since: pd.Series,
    history_hours: int,
    idx: int,
) -> np.ndarray | None:
    """
    Buduje wektor cech dla punktu o indeksie `idx`.

    Cechy:
      - pct z ostatnich `history_hours` godzin
      - hours_since_last_watering
      - sin oraz cos godziny dobowej
      - udział punktów interpolowanych w oknie
    """
    if idx < history_hours:
        return None

    pct = hourly["pct"].astype(float)
    filled = hourly["filled"].astype(bool) if "filled" in hourly.columns else None

    history = pct.iloc[idx - history_hours : idx]
    if history.isna().any():
        return None

    t = hourly.index[idx]
    h_since = hours_since.iloc[idx]
    if np.isnan(h_since):
        return None

    hour_of_day = t.hour + t.minute / 60.0
    angle = 2.0 * np.pi * hour_of_day / 24.0
    sin_h = np.sin(angle)
    cos_h = np.cos(angle)

    if filled is not None:
        fill_frac = float(filled.iloc[idx - history_hours : idx].mean())
    else:
        fill_frac = 0.0

    feats = np.concatenate([history.values, [h_since, sin_h, cos_h, fill_frac]])
    return feats


def build_ml_dataset(
    hourly: pd.DataFrame,
    cfg: dict,
    history_hours: int = 24,
    forecast_h: int = 1,
    warn_h: int = 6,
    waterings_iso: list[str] | None = None,
) -> dict:
    """
    Buduje zbiór danych do uczenia modeli ML.

    Zwraca słownik:
      {
        "X": ndarray [n_samples, n_features],
        "y_reg": ndarray [n_samples],
        "y_clf": ndarray [n_samples],
        "timestamps": ndarray[datetime64]
      }
    """
    if waterings_iso is None:
        waterings_iso = detect_waterings(hourly, jump_on=6.0)

    hourly = hourly.sort_index()
    hours_since = _compute_hours_since_last_watering(hourly, waterings_iso)
    pct = hourly["pct"].astype(float)

    n = len(hourly)
    if n < history_hours + max(forecast_h, warn_h) + 1:
        return {
            "X": np.zeros((0, history_hours + 4), dtype=float),
            "y_reg": np.zeros((0,), dtype=float),
            "y_clf": np.zeros((0,), dtype=int),
            "timestamps": np.array([], dtype="datetime64[ns]"),
        }

    X_list = []
    y_reg_list = []
    y_clf_list = []
    ts_list = []

    thresh = float(cfg.get("thresh_on", 70.0))

    for idx in range(history_hours, n - max(forecast_h, warn_h)):
        feats = _make_feature_vector_for_index(hourly, cfg, hours_since, history_hours, idx)
        if feats is None:
            continue

        # regresja: pct za forecast_h godzin
        target_idx_reg = idx + forecast_h
        y_reg_val = pct.iloc[target_idx_reg]
        if np.isnan(y_reg_val):
            continue

        # klasyfikacja: czy w ciągu warn_h godzin spadniemy poniżej progu
        fut = pct.iloc[idx + 1 : idx + 1 + warn_h]
        if len(fut) < warn_h or fut.isna().any():
            continue
        y_clf_val = 1 if (fut < thresh).any() else 0

        X_list.append(feats)
        y_reg_list.append(float(y_reg_val))
        y_clf_list.append(int(y_clf_val))
        ts_list.append(hourly.index[idx])

    if not X_list:
        return {
            "X": np.zeros((0, history_hours + 4), dtype=float),
            "y_reg": np.zeros((0,), dtype=float),
            "y_clf": np.zeros((0,), dtype=int),
            "timestamps": np.array([], dtype="datetime64[ns]"),
        }

    X = np.vstack(X_list)
    y_reg = np.array(y_reg_list, dtype=float)
    y_clf = np.array(y_clf_list, dtype=int)
    ts_arr = np.array(ts_list, dtype="datetime64[ns]")

    return {"X": X, "y_reg": y_reg, "y_clf": y_clf, "timestamps": ts_arr}


def load_ml_bundle() -> dict | None:
    """Wczytuje wytrenowane modele ML (jeśli istnieją)."""
    if joblib_load is None:
        # brak joblib – analytics działa, ale bez ML
        return None
    if not ML_CONFIG.exists():
        return None

    try:
        cfg_ml = json.loads(ML_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return None

    reg_name = cfg_ml.get("regressor")
    clf_name = cfg_ml.get("classifier")

    reg_model = None
    clf_model = None

    try:
        if reg_name:
            reg_path = ML_DIR / f"model_reg_{reg_name}.pkl"
            if reg_path.exists():
                reg_model = joblib_load(reg_path)
        if clf_name:
            clf_path = ML_DIR / f"model_clf_{clf_name}.pkl"
            if clf_path.exists():
                clf_model = joblib_load(clf_path)
    except Exception:
        # jeżeli coś pójdzie nie tak, po prostu nie używamy ML
        return None

    return {
        "config": cfg_ml,
        "reg_model": reg_model,
        "clf_model": clf_model,
    }


def make_ml_forecast(
    hourly: pd.DataFrame,
    cfg: dict,
    ml_bundle: dict,
    waterings_iso: list[str],
) -> dict | None:
    """
    Oblicza prostą prognozę ML dla ostatniego punktu serii.

    Zwraca m.in.:
      - predicted_pct_in_1h
      - will_cross_thresh_in_warn_window
      - probability_cross_thresh
    """
    if not ml_bundle:
        return None

    cfg_ml = ml_bundle.get("config", {})
    reg_model = ml_bundle.get("reg_model")
    clf_model = ml_bundle.get("clf_model")

    history_hours = int(cfg_ml.get("history_hours", 24))
    forecast_h = int(cfg_ml.get("forecast_h", 1))
    warn_h = int(cfg_ml.get("warn_h", 6))

    if len(hourly) < history_hours + 1:
        return None

    hourly = hourly.sort_index()
    hours_since = _compute_hours_since_last_watering(hourly, waterings_iso)

    idx = len(hourly) - 1  # ostatni dostępny punkt
    feats = _make_feature_vector_for_index(hourly, cfg, hours_since, history_hours, idx)
    if feats is None:
        return None

    X_now = feats.reshape(1, -1)
    thresh = float(cfg.get("thresh_on", 70.0))

    pred_reg = None
    if reg_model is not None:
        try:
            pred_reg = float(reg_model.predict(X_now)[0])
        except Exception:
            pred_reg = None

    pred_label = None
    prob1 = None
    if clf_model is not None:
        try:
            pred_label = int(clf_model.predict(X_now)[0])
            if hasattr(clf_model, "predict_proba"):
                prob = clf_model.predict_proba(X_now)[0]
                if len(prob) == 2:
                    prob1 = float(prob[1])
        except Exception:
            pred_label = None
            prob1 = None

    return {
        "regressor": cfg_ml.get("regressor"),
        "classifier": cfg_ml.get("classifier"),
        "history_hours": history_hours,
        "forecast_h": forecast_h,
        "warn_h": warn_h,
        "thresh_on": thresh,
        "predicted_pct_in_1h": pred_reg,
        "will_cross_thresh_in_warn_window": (bool(pred_label) if pred_label is not None else None),
        "probability_cross_thresh": prob1,
    }

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

    # blok ML – jeśli dostępne są wytrenowane modele, dodaj prognozę ML
    ml_bundle = load_ml_bundle()
    if ml_bundle is not None:
        ml_forecast = make_ml_forecast(hourly, cfg, ml_bundle, waterings)
        if ml_forecast is not None:
            forecast["ml"] = ml_forecast

    FORECAST_OUT.write_text(json.dumps(forecast, ensure_ascii=False), encoding="utf-8")
    save_series_week(hourly, waterings)
    print("[i] analytics: zapisano series_week.json i forecast.json")

if __name__ == "__main__":
    main()
