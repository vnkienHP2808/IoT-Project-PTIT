import json
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

RAW_CSV = "data/sensor_raw_60d.csv"
MODEL_PATH = Path("artifacts/model_xgb_rain60.pkl")
SCHEMA_PATH = Path("artifacts/feature_schema.json")

# Optional irrigation columns if có trong RAW
OPTIONAL_IRRIG_COLS = [
    "is_irrigating_now", "irrig_total_min_last_3h", "irrig_total_min_last_6h"
]

def build_features_window(df: pd.DataFrame, device_id="esp32-01", now_ts=None):
    df = df[df["device_id"] == device_id].sort_values("ts")
    if now_ts is None:
        now_ts = df["ts"].max()
    window = df[df["ts"].between(now_ts - pd.Timedelta(minutes=120), now_ts)]
    if len(window) < 12:
        raise ValueError("Not enough rows in 120-minute window")

    last = window.iloc[-1]
    lag = lambda s, k: s.iloc[-1 - k]
    mean_last = lambda s, k: s.iloc[-k:].mean()

    feats = {
        "temp_c": last["temp_c"],
        "rh_pct": last["rh_pct"],
        "pressure_hpa": last["pressure_hpa"],
        "soil_moist_pct": last["soil_moist_pct"],
        "rain_mm_5min": last["rain_mm_5min"],
        "pressure_delta15": last["pressure_hpa"] - lag(window["pressure_hpa"], 3),
        "rh_delta15":       last["rh_pct"]       - lag(window["rh_pct"], 3),
        "temp_delta15":     last["temp_c"]       - lag(window["temp_c"], 3),
        "temp_c_lag15":           lag(window["temp_c"], 3),
        "rh_pct_lag15":           lag(window["rh_pct"], 3),
        "pressure_hpa_lag15":     lag(window["pressure_hpa"], 3),
        "soil_moist_pct_lag15":   lag(window["soil_moist_pct"], 3),
        "temp_c_mean30":          mean_last(window["temp_c"], 6),
        "rh_pct_mean30":          mean_last(window["rh_pct"], 6),
        "pressure_hpa_mean30":    mean_last(window["pressure_hpa"], 6),
        "soil_moist_pct_mean30":  mean_last(window["soil_moist_pct"], 6),
        "rain_in_last_15m": 1 if window["rain_mm_5min"].iloc[-3:].sum() > 0 else 0,
        "hour_of_day": last["ts"].hour,
        "day_of_week": last["ts"].dayofweek,
    }

    # optional irrigation features (nếu có trong RAW thì dùng)
    for c in OPTIONAL_IRRIG_COLS:
        if c in window.columns:
            feats[c] = float(window[c].iloc[-1])

    return feats, str(last["ts"]), device_id

if __name__ == "__main__":
    now_arg = sys.argv[1] if len(sys.argv) > 1 else None
    raw = pd.read_csv(RAW_CSV, parse_dates=["ts"])
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        FEATURES = json.load(f)["features"]
    model = joblib.load(MODEL_PATH)

    feats, ts_str, dev = build_features_window(
        raw, device_id="esp32-01",
        now_ts=pd.to_datetime(now_arg) if now_arg else None
    )
    x = np.float32([feats.get(k, 0.0) for k in FEATURES]).reshape(1, -1)
    p = float(model.predict_proba(x)[0, 1])

    print(json.dumps({
        "ts": ts_str,
        "device_id": dev,
        "p_rain_60": round(p, 4),
        "model": MODEL_PATH.name
    }, ensure_ascii=False))
