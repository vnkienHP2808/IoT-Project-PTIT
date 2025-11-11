# src/app.py — Flask demo UI with proba-based suggestions
# Run from project root:
#   set PYTHONPATH=%CD%
#   python -m src.app
# Or run inside src/:
#   python app.py

import os, sys, json
from pathlib import Path
from flask import Flask, request, render_template_string, jsonify
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import joblib

# --- Ensure project root on sys.path (so `src.*` works when running from src/) ---
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.wrappers import XGBBoosterWithThreshold  # noqa: E402

load_dotenv()

APP_TITLE = "Rain Nowcast — Demo"
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

SENSOR_CSV = DATA_DIR / "sensor_raw_60d.csv"
LBL_FIXED  = DATA_DIR / "labels_rain_60d_fixed.csv"
LABELS_CSV = LBL_FIXED if LBL_FIXED.exists() else (DATA_DIR / "labels_rain_60d.csv")

# ===== Load data (force UTC to avoid tz-aware vs tz-naive issues) =====
sensor = pd.read_csv(SENSOR_CSV, parse_dates=["ts"]).sort_values(["device_id","ts"]).reset_index(drop=True)
labels = pd.read_csv(LABELS_CSV, parse_dates=["ts"]).sort_values(["device_id","ts"]).reset_index(drop=True)

sensor["ts"] = pd.to_datetime(sensor["ts"], utc=True)
labels["ts"] = pd.to_datetime(labels["ts"], utc=True)

use_cols = ["ts","device_id"]
if "rain_next_60" in labels.columns:
    use_cols.append("rain_next_60")
if "rain_amount_next_60_mm" in labels.columns:
    use_cols.append("rain_amount_next_60_mm")

df = sensor.merge(labels[use_cols], on=["ts","device_id"], how="inner")

# ===== Load model & metadata =====
META_PATH = MODEL_DIR / "metadata.json"
with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)
FEATURES = META["features"]
THR_DEF  = float(META.get("threshold_default", 0.5))

CLS_PATH = MODEL_DIR / "xgb_nowcast.pkl"
REG_PATH = MODEL_DIR / "xgb_amount.pkl"

# --- load nowcast model (wrapper or raw payload) ---
def load_nowcast_model(path: Path):
    obj = joblib.load(path)
    if hasattr(obj, "predict_proba") and hasattr(obj, "predict"):
        return obj, "wrapper"
    if isinstance(obj, dict) and ("booster_bytes" in obj or "booster" in obj):
        import xgboost as xgb
        if "booster_bytes" in obj:
            bst = xgb.Booster()
            bst.load_model(obj["booster_bytes"])
        else:
            bst = obj["booster"]
        thr = float(obj.get("threshold", THR_DEF))
        class _RawNowcast:
            def __init__(self, booster, threshold):
                self._booster = booster
                self.threshold = threshold
            def predict_proba(self, X):
                dm = xgb.DMatrix(X)
                best_iter = getattr(self._booster, "best_iteration", None)
                if best_iter is not None and best_iter >= 0:
                    p = self._booster.predict(dm, iteration_range=(0, best_iter + 1))
                else:
                    p = self._booster.predict(dm)
                p = np.asarray(p).reshape(-1)
                return np.c_[1-p, p]
            def predict(self, X, threshold=None):
                th = self.threshold if threshold is None else float(threshold)
                return (self.predict_proba(X)[:,1] >= th).astype(int)
        return _RawNowcast(bst, thr), "raw"
    raise RuntimeError("Unsupported nowcast model format.")

model, model_kind = load_nowcast_model(CLS_PATH)

# regression (optional)
reg_model = joblib.load(REG_PATH) if REG_PATH.exists() else None

# ===== Feature vector builder =====

def row_to_feature_vector(row: pd.Series) -> np.ndarray:
    x = np.zeros((1, len(FEATURES)), dtype="float32")
    for i, name in enumerate(FEATURES):
        v = row[name] if name in row.index else 0.0
        try:
            x[0, i] = float(v)
        except Exception:
            x[0, i] = 0.0
    return x

# ===== Suggestion by model proba (per device) =====

def top_prob_rainy_slots(df_device: pd.DataFrame, topn: int = 10):
    rows = []
    for _, r in df_device.iterrows():
        x = row_to_feature_vector(r)
        p = float(model.predict_proba(x)[0, 1])
        rows.append((r["ts"], p))
    # sort by proba desc
    rows.sort(key=lambda t: t[1], reverse=True)
    # return list of dicts
    out = []
    for ts, p in rows[:topn]:
        out.append({
            "ts": ts.strftime("%Y-%m-%dT%H:%M"),  # ISO minute
            "prob": round(p, 4)
        })
    return out

# ===== Flask app =====
app = Flask(__name__)

TPL = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{{title}}</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    .row { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
    label { font-weight: 600; margin-right: 8px; }
    input, select { padding: 6px 8px; }
    .btn { padding: 8px 14px; border: 0; border-radius: 8px; background: #111827; color: white; cursor: pointer; }
    .muted { color: #6b7280; }
    table { border-collapse: collapse; width: 100%; }
    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee; }
    .ok { color: #16a34a; font-weight: 700; }
    .warn { color: #dc2626; font-weight: 700; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f3f4f6; }
    .suggest { display: inline-flex; gap: 8px; align-items: center; }
    .suggest button { padding: 4px 10px; border: 1px solid #ddd; border-radius: 8px; background: white; cursor: pointer; }
  </style>
</head>
<body>
  <h1>{{title}}</h1>

  <div class=\"card\">
    <form method=\"GET\" action=\"/\">
      <div class=\"row\">
        <label for=\"ts\">Chọn thời điểm (từ dữ liệu demo):</label>
        <input type=\"datetime-local\" id=\"ts\" name=\"ts\" value=\"{{ts_value}}\" step=\"300\" />
        <label for=\"device\">Thiết bị:</label>
        <select id=\"device\" name=\"device\">
          {% for d in devices %}
            <option value=\"{{d}}\" {% if d==device %}selected{% endif %}>{{d}}</option>
          {% endfor %}
        </select>
        <button class=\"btn\" type=\"submit\">Lấy bản ghi gần nhất</button>
      </div>
      <p class=\"muted\">Tip: chọn ngày/giờ quá khứ → hệ thống tìm bản ghi gần nhất ≤ thời điểm đó và chạy infer.</p>
    </form>
  </div>

  <div class=\"card\">
    <h3>Gợi ý mốc \"khả năng mưa cao\" theo model</h3>
    <div class=\"row\">
      <div class=\"suggest\">
        <label for=\"n\">Số gợi ý</label>
        <input id=\"n\" type=\"number\" value=\"10\" min=\"1\" max=\"50\" style=\"width:80px\"/>
        <button onclick=\"loadSuggest(); return false;\">Lấy gợi ý</button>
      </div>
    </div>
    <div id=\"sugList\"></div>
  </div>

  {% if has_row %}
  <div class=\"card\">
    <h3>Bản ghi được chọn</h3>
    <table>
      <tr><th>Thời điểm (UTC)</th><td>{{ row.ts }}</td></tr>
      <tr><th>Thiết bị</th><td>{{ row.device_id }}</td></tr>
      <tr><th>Độ ẩm đất</th><td>{{ row.soil_moist_pct }}</td></tr>
      <tr><th>Mưa 5'</th><td>{{ row.rain_mm_5min }}</td></tr>
      {% if has_label %}
      <tr><th>Nhãn mưa 60'</th><td><span class=\"pill\">{{ row.rain_next_60 }}</span></td></tr>
      {% endif %}
    </table>
  </div>

  <div class=\"card\">
    <h3>Dự báo mưa 60 phút tới</h3>
    {% if pred.label == 1 %}
      <p class=\"warn\">Có khả năng mưa (label=1) — p={{\"%.3f\" % pred.prob}} (thr={{\"%.3f\" % pred.thr}})</p>
    {% else %}
      <p class=\"ok\">Ít khả năng mưa (label=0) — p={{\"%.3f\" % pred.prob}} (thr={{\"%.3f\" % pred.thr}})</p>
    {% endif %}
    {% if amount is not none %}
      <p>Lượng mưa ước tính 60' tới: <b>{{\"%.2f\" % amount}} mm</b></p>
    {% endif %}
  </div>

  <div class=\"card\">
    <h3>Quyết định tưới (demo)</h3>
    {% if pred.label == 1 %}
      <p class=\"warn\">Đề xuất: <b>Hoãn tưới</b> do rủi ro mưa cao.</p>
    {% else %}
      <p class=\"ok\">Đề xuất: <b>Tưới theo lịch</b>.</p>
    {% endif %}
  </div>
  {% endif %}

<script>
async function loadSuggest(){
  const device = document.getElementById('device').value;
  const n = document.getElementById('n').value || 10;
  const res = await fetch(`/suggest?device=${encodeURIComponent(device)}&n=${encodeURIComponent(n)}`);
  const data = await res.json();
  const box = document.getElementById('sugList');
  if(!Array.isArray(data) || data.length === 0){ box.innerHTML = '<p class="muted">Không có gợi ý.</p>'; return; }
  let html = '<table><tr><th>#</th><th>Timestamp</th><th>p(rain)</th><th></th></tr>';
  data.forEach((it,idx)=>{
    html += `<tr><td>${idx+1}</td><td>${it.ts}</td><td>${it.prob}</td>`+
            `<td><button onclick=\"applyTs('${it.ts}')\">Chọn</button></td></tr>`
  });
  html += '</table>';
  box.innerHTML = html;
}
function applyTs(ts){
  document.getElementById('ts').value = ts;
}
</script>
</body>
</html>
"""

LOCAL_TZ = "Asia/Ho_Chi_Minh"

@app.route("/")
def home():
    q_ts = request.args.get("ts")
    device = request.args.get("device", default=str(df["device_id"].iloc[0]))

    if not q_ts:
        mid_idx = len(df) // 2
        q_ts = df.iloc[mid_idx]["ts"].to_pydatetime().strftime("%Y-%m-%dT%H:%M")

    ts_query = pd.to_datetime(q_ts)
    if ts_query.tzinfo is None:
        ts_query = ts_query.tz_localize(LOCAL_TZ).tz_convert("UTC")
    else:
        ts_query = ts_query.tz_convert("UTC")

    sub = df[df["device_id"] == device]
    chosen = sub[sub["ts"] <= ts_query].tail(1)
    if chosen.empty:
        idx = (sub["ts"] - ts_query).abs().idxmin()
        chosen = sub.loc[[idx]]

    row_s = chosen.iloc[0]
    row_dict = row_s.to_dict()
    try:
        row_dict["ts"] = row_s["ts"].strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        pass

    x = row_to_feature_vector(row_s)
    p = float(model.predict_proba(x)[0, 1])
    label = 1 if p >= THR_DEF else 0

    amount = None
    if reg_model is not None:
        try:
            import xgboost as xgb
            if isinstance(reg_model, xgb.Booster):
                amount = float(max(0.0, reg_model.predict(xgb.DMatrix(x))[0]))
            else:
                amount = float(max(0.0, reg_model.predict(x)[0]))
        except Exception:
            amount = None

    devices = list(df["device_id"].unique())
    return render_template_string(
        TPL,
        title=APP_TITLE,
        devices=devices,
        device=device,
        ts_value=q_ts,
        row=row_dict,
        has_row=True,
        has_label=("rain_next_60" in row_dict),
        pred={"prob": p, "label": label, "thr": THR_DEF},
        amount=amount,
    )

@app.route("/suggest")
def suggest():
    device = request.args.get("device", default=str(df["device_id"].iloc[0]))
    try:
        topn = int(request.args.get("n", 10))
    except Exception:
        topn = 10

    sub = df[df["device_id"] == device]
    if sub.empty:
        return jsonify([])

    suggestions = top_prob_rainy_slots(sub, topn=topn)
    return jsonify(suggestions)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5050)), debug=True)
