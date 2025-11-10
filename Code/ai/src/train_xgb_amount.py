# src/train_xgb_amount.py
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "sensor_raw_60d.csv"
LBL_FIXED = DATA_DIR / "labels_rain_60d_fixed.csv"   # dùng file fixed nếu có
LBL_CSV = LBL_FIXED if LBL_FIXED.exists() else (DATA_DIR / "labels_rain_60d.csv")

def build_dataset():
    raw = pd.read_csv(RAW_CSV, parse_dates=["ts"]).sort_values(["device_id","ts"]).reset_index(drop=True)
    lbl = pd.read_csv(LBL_CSV, parse_dates=["ts"])
    # cần cột 'rain_amount_next_60_mm' trong labels_* (nếu chưa có, bạn có thể tự tổng 12 bước 5' tương lai)
    if "rain_amount_next_60_mm" not in lbl.columns:
        raise ValueError("labels_* thiếu cột 'rain_amount_next_60_mm'.")

    df = raw.merge(lbl[["ts","device_id","rain_amount_next_60_mm"]], on=["ts","device_id"], how="inner")
    df = df.sort_values(["device_id","ts"]).reset_index(drop=True)

    # === Feature engineering (giống head phân loại để đồng bộ) ===
    def add_feats(g: pd.DataFrame):
        g = g.copy()
        for col in ["temp_c","rh_pct","pressure_hpa","soil_moist_pct"]:
            g[f"{col}_lag15"]  = g[col].shift(3)
            g[f"{col}_mean30"] = g[col].rolling(6).mean()
        g["pressure_delta15"] = g["pressure_hpa"] - g["pressure_hpa"].shift(3)
        g["rh_delta15"]       = g["rh_pct"]        - g["rh_pct"].shift(3)
        g["temp_delta15"]     = g["temp_c"]        - g["temp_c"].shift(3)
        g["rain_in_last_15m"] = g["rain_mm_5min"].rolling(3).sum().gt(0).astype(int)
        g["hour_of_day"]      = g["ts"].dt.hour
        g["day_of_week"]      = g["ts"].dt.dayofweek
        return g

    df = df.groupby("device_id", group_keys=False).apply(add_feats).dropna().reset_index(drop=True)

    FEATURES = [
        "temp_c","rh_pct","pressure_hpa","soil_moist_pct","rain_mm_5min",
        "pressure_delta15","rh_delta15","temp_delta15",
        "temp_c_lag15","rh_pct_lag15","pressure_hpa_lag15","soil_moist_pct_lag15",
        "temp_c_mean30","rh_pct_mean30","pressure_hpa_mean30","soil_moist_pct_mean30",
        "rain_in_last_15m","hour_of_day","day_of_week",
    ]
    X = df[FEATURES].astype("float32").values
    y = df["rain_amount_next_60_mm"].astype("float32").values.clip(min=0)  # mm >= 0
    return df, X, y, FEATURES

def main():
    df, X, y, FEATURES = build_dataset()

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, shuffle=False)

    dtrain, dvalid = xgb.DMatrix(Xtr, label=ytr), xgb.DMatrix(Xte, label=yte)

    params = {
        "objective": "reg:squarederror",   # hồi quy
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
    }

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1200,
        evals=[(dtrain,"train"), (dvalid,"valid")],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    # === Evaluate ===
    pred = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    mae = mean_absolute_error(yte, pred)

    # một số bản sklearn cũ không có squared=
    # rmse = mean_squared_error(yte, pred, squared=False)  # <-- gây lỗi ở máy bạn
    rmse = np.sqrt(mean_squared_error(yte, pred))          # <-- dùng cách này, chạy mọi phiên bản

    r2 = r2_score(yte, pred)
    print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")


    # === Save model & metadata ===
    joblib.dump(bst, MODEL_DIR / "xgb_amount.pkl")
    meta_amount = {
        "features": FEATURES,
        "target": "rain_amount_next_60_mm",
        "note": "XGBoost regression for rainfall amount in next 60 minutes"
    }
    with open(MODEL_DIR / "metadata_amount.json", "w", encoding="utf-8") as f:
        json.dump(meta_amount, f, ensure_ascii=False, indent=2)

    print("Saved models/xgb_amount.pkl & models/metadata_amount.json")

if __name__ == "__main__":
    main()
