from __future__ import annotations

"""
Offline'owy skrypt do trenowania prostych modeli ML na danych z GabiPlant.

Użycie (w katalogu plant-analytics):
    python ml_training.py
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# sklearn – wymagane w środowisku uruchomieniowym (nie w loggerze na ESP itp.)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
)
from joblib import dump

# Importujemy narzędzia z analytics – żeby mieć jedną logikę preprocessingu
from analytics import (
    DATA_DIR,
    read_config,
    load_csv_last_days,
    hourly_series,
    detect_waterings,
    build_ml_dataset,
)

ML_DIR = (DATA_DIR / "ml").resolve()
ML_CONFIG = ML_DIR / "ml_config.json"
ML_METRICS = ML_DIR / "ml_metrics.json"


def train_models(
    days: int = 7,
    history_hours: int = 24,
    forecast_h: int = 1,
    warn_h: int = 6,
    train_frac: float = 0.7,
) -> None:
    """Główna funkcja – trenuje modele i zapisuje je do plików w DATA_DIR/ml."""
    ML_DIR.mkdir(parents=True, exist_ok=True)

    cfg = read_config()
    df = load_csv_last_days(days=days)
    if df.empty:
        print("[!] Brak danych z loggera – nie można trenować modeli.")
        return

    hourly = hourly_series(df)
    waterings = detect_waterings(hourly, jump_on=6.0)

    dataset = build_ml_dataset(
        hourly=hourly,
        cfg=cfg,
        history_hours=history_hours,
        forecast_h=forecast_h,
        warn_h=warn_h,
        waterings_iso=waterings,
    )

    X = dataset["X"]
    y_reg = dataset["y_reg"]
    y_clf = dataset["y_clf"]
    timestamps = dataset["timestamps"]

    n_samples = X.shape[0]
    if n_samples < 30:
        print(f"[!] Za mało przykładów ({n_samples}) do sensownego trenowania modeli.")
        return

    # --- podział train/test po czasie ---
    idx_split = int(len(timestamps) * train_frac)
    X_train, X_test = X[:idx_split], X[idx_split:]
    y_reg_train, y_reg_test = y_reg[:idx_split], y_reg[idx_split:]
    y_clf_train, y_clf_test = y_clf[:idx_split], y_clf[idx_split:]

    # --- REGRESJA ---
    regressors = {
        "linear": LinearRegression(),
        "tree": DecisionTreeRegressor(random_state=42),
        "rf": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(32,),
            activation="relu",
            max_iter=500,
            random_state=42,
        ),
    }

    reg_metrics: dict[str, dict[str, float]] = {}
    reg_models_trained: dict[str, object] = {}

    for name, model in regressors.items():
        model.fit(X_train, y_reg_train)
        y_pred_test = model.predict(X_test)

        mae = float(mean_absolute_error(y_reg_test, y_pred_test))
        mse = mean_squared_error(y_reg_test, y_pred_test)
        rmse = float(np.sqrt(mse))

        reg_metrics[name] = {"mae": mae, "rmse": rmse}
        reg_models_trained[name] = model

        print(f"[reg] {name}: MAE={mae:.3f}, RMSE={rmse:.3f}")

    # wybierz najlepszy po RMSE
    best_reg_name = min(reg_metrics, key=lambda k: reg_metrics[k]["rmse"])
    best_reg_model = reg_models_trained[best_reg_name]
    print(f"[reg] najlepszy model: {best_reg_name}")

    # --- KLASYFIKACJA ---
    classifiers = {
        "tree": DecisionTreeClassifier(random_state=42),
        "rf": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            max_iter=500,
            random_state=42,
        ),
    }

    clf_metrics: dict[str, dict[str, float]] = {}
    clf_models_trained: dict[str, object] = {}

    for name, model in classifiers.items():
        model.fit(X_train, y_clf_train)
        y_pred_test = model.predict(X_test)

        acc = float(accuracy_score(y_clf_test, y_pred_test))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_clf_test, y_pred_test, average="binary", zero_division=0
        )

        clf_metrics[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        clf_models_trained[name] = model

        print(
            f"[clf] {name}: acc={acc:.3f}, prec={prec:.3f}, "
            f"recall={rec:.3f}, f1={f1:.3f}"
        )

    best_clf_name = max(clf_metrics, key=lambda k: clf_metrics[k]["f1"])
    best_clf_model = clf_models_trained[best_clf_name]
    print(f"[clf] najlepszy model (po F1): {best_clf_name}")

    # --- zapis modeli i metadanych ---
    reg_path = ML_DIR / f"model_reg_{best_reg_name}.pkl"
    clf_path = ML_DIR / f"model_clf_{best_clf_name}.pkl"

    dump(best_reg_model, reg_path)
    dump(best_clf_model, clf_path)

    config_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "history_hours": history_hours,
        "forecast_h": forecast_h,
        "warn_h": warn_h,
        "regressor": best_reg_name,
        "classifier": best_clf_name,
        "n_samples": int(n_samples),
    }
    ML_CONFIG.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics_payload = {
        "regression": reg_metrics,
        "classification": clf_metrics,
        "train_frac": train_frac,
        "n_samples": int(n_samples),
    }
    ML_METRICS.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[i] Zapisano najlepszy model regresji do {reg_path.name}")
    print(f"[i] Zapisano najlepszy model klasyfikacji do {clf_path.name}")
    print(f"[i] Zapisano ml_config.json oraz ml_metrics.json w {ML_DIR}")


if __name__ == "__main__":
    train_models()
