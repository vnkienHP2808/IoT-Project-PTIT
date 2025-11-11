# src/train_xgb.py
import os, json, argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, classification_report, confusion_matrix

# ====== paths ======
ROOT = Path(__file__).resolve().parents[1]  # về thư mục project
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV   = DATA_DIR / "sensor_raw_60d.csv"
LBL_CSV   = DATA_DIR / "labels_rain_60d_fixed.csv"  # dùng file fixed nếu có, không thì labels_rain_60d.csv
LBL_CSV   = LBL_CSV if LBL_CSV.exists() else (DATA_DIR / "labels_rain_60d.csv")

# ====== feature engineering (rút gọn giống notebook, có thể chỉnh theo bản của bạn) ======
def build_dataset():
    raw = pd.read_csv(RAW_CSV, parse_dates=["ts"]).sort_values(["device_id","ts"]).reset_index(drop=True)
    lbl = pd.read_csv(LBL_CSV, parse_dates=["ts"])

    df = raw.merge(lbl[["ts","device_id","rain_next_60"]], on=["ts","device_id"], how="inner")
    df = df.sort_values(["device_id","ts"]).reset_index(drop=True)

    def add_feats(g: pd.DataFrame) -> pd.DataFrame:
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

    df = df.groupby("device_id", group_keys=False).apply(add_feats)
    df = df.dropna().reset_index(drop=True)

    FEATURES = [
        "temp_c","rh_pct","pressure_hpa","soil_moist_pct","rain_mm_5min",
        "pressure_delta15","rh_delta15","temp_delta15",
        "temp_c_lag15","rh_pct_lag15","pressure_hpa_lag15","soil_moist_pct_lag15",
        "temp_c_mean30","rh_pct_mean30","pressure_hpa_mean30","soil_moist_pct_mean30",
        "rain_in_last_15m","hour_of_day","day_of_week",
    ]
    X = df[FEATURES].astype("float32").values
    y = df["rain_next_60"].astype(int).values
    return df, X, y, FEATURES

def train_and_save(save_mode: str = "wrapper"):
    from wrappers import XGBBoosterWithThreshold  # import từ module cố định

    df, X, y, FEATURES = build_dataset()

    pos, neg = (y==1).sum(), (y==0).sum()
    scale_pos_weight = float(neg) / max(1.0, float(pos))

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, shuffle=False)
    dtrain, dvalid = xgb.DMatrix(Xtr, label=ytr), xgb.DMatrix(Xte, label=yte)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.03,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
    }

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1200,
        evals=[(dtrain,"train"), (dvalid,"valid")],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    # ===== evaluate quickly =====
    proba = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
    auc   = roc_auc_score(yte, proba)
    prauc = average_precision_score(yte, proba)
    pred  = (proba >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary")
    print(f"AUC-ROC: {auc:.4f} | PR-AUC: {prauc:.4f}")
    print(f"@0.50  Acc: {(pred==yte).mean():.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
    print("--- Confusion ---")
    print(confusion_matrix(yte, pred))
    print(classification_report(yte, pred, digits=4))

    # ===== choose threshold (ví dụ theo F1) =====
    thr_grid = np.linspace(0.1, 0.9, 33)
    best_thr, best_f1 = 0.5, -1
    for th in thr_grid:
        pr = (proba >= th).astype(int)
        _, _, f1_, _ = precision_recall_fscore_support(yte, pr, average="binary", zero_division=0)
        if f1_ > best_f1:
            best_f1, best_thr = f1_, float(th)
    print(f"Chosen threshold (best-F1): {best_thr:.3f} (F1={best_f1:.4f})")

    # ===== save metadata =====
    meta = {
        "features": FEATURES,
        "threshold_default": best_thr
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ===== save model =====
    if save_mode == "wrapper":
        model = XGBBoosterWithThreshold(bst, threshold=best_thr)
        joblib.dump(model, MODEL_DIR / "xgb_nowcast.pkl")
        print("Saved models/xgb_nowcast.pkl (wrapper) + models/metadata.json")

    elif save_mode == "raw":
        payload = {
            "booster_bytes": bst.save_raw(),
            "best_iteration": int(getattr(bst, "best_iteration", -1)),
            "threshold": best_thr
        }
        joblib.dump(payload, MODEL_DIR / "xgb_nowcast.pkl")
        print("Saved models/xgb_nowcast.pkl (raw payload) + models/metadata.json")

    else:
        raise ValueError("save_mode must be 'wrapper' or 'raw'")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-mode", choices=["wrapper","raw"], default="wrapper",
                    help="wrapper: lưu XGBBoosterWithThreshold | raw: lưu booster_bytes+threshold")
    args = ap.parse_args()
    train_and_save(args.save_mode)
