# ROADMAP BỔ SUNG TÍNH NĂNG - DỰ ÁN IoT TƯỚI CÂY

## 📋 TÓM TẮT HIỆN TRẠNG

### ✅ Đã có:
- ✅ Model dự đoán mưa 60 phút (phân loại + hồi quy)
- ✅ Feature engineering từ cảm biến
- ✅ Flask web UI để demo
- ✅ CLI inference
- ✅ Dữ liệu 60 ngày quá khứ

### ❌ Còn thiếu:
- ❌ Model dự đoán mưa 30 phút
- ❌ Dự báo thời tiết 3-7 ngày (cần API bên ngoài)
- ❌ Thuật toán lập lịch tưới thông minh
- ❌ Mô hình nhu cầu nước của cây
- ❌ Tối ưu hóa lịch tưới

---

## 🎯 ROADMAP TRIỂN KHAI (Chia làm 3 GIAI ĐOẠN)

---

# GIAI ĐOẠN 1: BỔ SUNG NOWCAST 30 PHÚT (1-2 ngày) ⭐ ƯU TIÊN CAO

## Mục tiêu:
Có thêm model dự đoán mưa **30 phút** tới, tăng độ chính xác cho quyết định tưới ngắn hạn.

## Các bước thực hiện:

### 1.1. Tạo nhãn mưa 30 phút
**File**: `scripts/create_labels_30min.py`

```python
import pandas as pd
from pathlib import Path

def create_rain_30min_labels():
    """Tạo nhãn rain_next_30 từ dữ liệu sensor"""
    DATA_DIR = Path("data")
    sensor = pd.read_csv(DATA_DIR / "sensor_raw_60d.csv", parse_dates=["ts"])
    sensor = sensor.sort_values(["device_id", "ts"]).reset_index(drop=True)
    
    def add_labels(g):
        g = g.copy()
        # 30 phút = 6 bước 5 phút
        g["rain_next_30"] = g["rain_mm_5min"].shift(-6).rolling(6).sum().gt(0).astype(int)
        g["rain_amount_next_30_mm"] = g["rain_mm_5min"].shift(-6).rolling(6).sum()
        return g
    
    labels = sensor.groupby("device_id", group_keys=False).apply(add_labels)
    labels = labels[["ts", "device_id", "rain_next_30", "rain_amount_next_30_mm"]]
    labels = labels.dropna()
    
    labels.to_csv(DATA_DIR / "labels_rain_30d.csv", index=False)
    print(f"✓ Đã tạo {len(labels)} nhãn 30 phút")
    print(f"  - Tỷ lệ mưa: {labels['rain_next_30'].mean():.2%}")

if __name__ == "__main__":
    create_rain_30min_labels()
```

**Chạy**: `python scripts/create_labels_30min.py`

---

### 1.2. Train model nowcast 30 phút
**File**: `src/train_xgb_nowcast_30min.py`

```python
# Tương tự train_xgb_nowcast.py, chỉ thay đổi:
# - Đọc labels_rain_30d.csv
# - Target: rain_next_30
# - Lưu: models/xgb_nowcast_30min.pkl
# - Metadata: models/metadata_30min.json

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_fscore_support, classification_report, confusion_matrix
)
from wrappers import XGBBoosterWithThreshold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

RAW_CSV = DATA_DIR / "sensor_raw_60d.csv"
LBL_CSV = DATA_DIR / "labels_rain_30d.csv"  # ← Thay đổi

def build_dataset():
    raw = pd.read_csv(RAW_CSV, parse_dates=["ts"]).sort_values(["device_id","ts"]).reset_index(drop=True)
    lbl = pd.read_csv(LBL_CSV, parse_dates=["ts"])
    
    df = raw.merge(lbl[["ts","device_id","rain_next_30"]], on=["ts","device_id"], how="inner")  # ← Thay đổi
    df = df.sort_values(["device_id","ts"]).reset_index(drop=True)
    
    def add_feats(g):
        g = g.copy()
        for col in ["temp_c","rh_pct","pressure_hpa","soil_moist_pct"]:
            g[f"{col}_lag15"] = g[col].shift(3)
            g[f"{col}_mean30"] = g[col].rolling(6).mean()
        g["pressure_delta15"] = g["pressure_hpa"] - g["pressure_hpa"].shift(3)
        g["rh_delta15"] = g["rh_pct"] - g["rh_pct"].shift(3)
        g["temp_delta15"] = g["temp_c"] - g["temp_c"].shift(3)
        g["rain_in_last_15m"] = g["rain_mm_5min"].rolling(3).sum().gt(0).astype(int)
        g["hour_of_day"] = g["ts"].dt.hour
        g["day_of_week"] = g["ts"].dt.dayofweek
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
    y = df["rain_next_30"].astype(int).values  # ← Thay đổi
    return df, X, y, FEATURES

def main():
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
    
    # Evaluate
    proba = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
    auc = roc_auc_score(yte, proba)
    prauc = average_precision_score(yte, proba)
    pred = (proba >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary")
    print(f"30-min Nowcast | AUC: {auc:.4f} PR-AUC: {prauc:.4f}")
    print(f"@0.50 Acc: {(pred==yte).mean():.4f} Prec: {prec:.4f} Rec: {rec:.4f} F1: {f1:.4f}")
    print(confusion_matrix(yte, pred))
    
    # Find best threshold
    thr_grid = np.linspace(0.1, 0.9, 33)
    best_thr, best_f1 = 0.5, -1
    for th in thr_grid:
        pr = (proba >= th).astype(int)
        _, _, f1_, _ = precision_recall_fscore_support(yte, pr, average="binary", zero_division=0)
        if f1_ > best_f1:
            best_f1, best_thr = f1_, float(th)
    print(f"Best threshold: {best_thr:.3f} (F1={best_f1:.4f})")
    
    # Save
    meta = {"features": FEATURES, "threshold_default": best_thr}
    with open(MODEL_DIR / "metadata_30min.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    model = XGBBoosterWithThreshold(bst, threshold=best_thr)
    joblib.dump(model, MODEL_DIR / "xgb_nowcast_30min.pkl")
    print("✓ Saved models/xgb_nowcast_30min.pkl")

if __name__ == "__main__":
    main()
```

**Chạy**: `python src/train_xgb_nowcast_30min.py`

---

### 1.3. Tương tự train model amount 30 phút
**File**: `src/train_xgb_amount_30min.py`
- Target: `rain_amount_next_30_mm`
- Save: `models/xgb_amount_30min.pkl`

---

# GIAI ĐOẠN 2: TÍCH HỢP DỰ BÁO THỜI TIẾT 3-7 NGÀY (2-3 ngày) ⭐⭐

## Mục tiêu:
Lấy dữ liệu dự báo thời tiết từ API bên ngoài (OpenWeatherMap/AccuWeather) để có thông tin mưa 3-7 ngày tới.

## 2.1. Chọn và đăng ký API

### Khuyến nghị: **OpenWeatherMap (OWM)**
- ✅ Free tier: 1000 calls/day
- ✅ Dự báo 5 ngày/3 giờ
- ✅ Dễ tích hợp
- 🔗 [https://openweathermap.org/api](https://openweathermap.org/api)

**Đăng ký:**
1. Tạo tài khoản tại https://home.openweathermap.org/users/sign_up
2. Lấy API key tại https://home.openweathermap.org/api_keys
3. Lưu vào file `.env`:
   ```
   OWM_API_KEY=your_api_key_here
   ```

---

## 2.2. Module tích hợp API thời tiết
**File**: `src/weather_forecast.py`

```python
"""
Module lấy dự báo thời tiết từ OpenWeatherMap
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

class WeatherForecastAPI:
    """Lấy dự báo thời tiết 5 ngày từ OpenWeatherMap"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OWM_API_KEY")
        if not self.api_key:
            raise ValueError("OWM_API_KEY not found in environment")
        self.base_url = "https://api.openweathermap.org/data/2.5/forecast"
    
    def get_forecast(self, lat: float, lon: float) -> pd.DataFrame:
        """
        Lấy dự báo 5 ngày/3 giờ từ OWM
        
        Args:
            lat: Vĩ độ (ví dụ: 10.762622 cho TP.HCM)
            lon: Kinh độ (ví dụ: 106.660172 cho TP.HCM)
        
        Returns:
            DataFrame với các cột:
            - ts: timestamp
            - temp_c: nhiệt độ (°C)
            - humidity_pct: độ ẩm (%)
            - pressure_hpa: áp suất (hPa)
            - rain_prob: xác suất mưa (0-1)
            - rain_3h_mm: lượng mưa 3h (mm)
            - clouds_pct: độ che phủ mây (%)
            - wind_mps: tốc độ gió (m/s)
        """
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",  # Celsius
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"❌ Lỗi khi gọi API: {e}")
            return pd.DataFrame()
        
        # Parse forecast
        forecasts = []
        for item in data.get("list", []):
            ts = pd.to_datetime(item["dt"], unit="s", utc=True)
            forecasts.append({
                "ts": ts,
                "temp_c": item["main"]["temp"],
                "humidity_pct": item["main"]["humidity"],
                "pressure_hpa": item["main"]["pressure"],
                "rain_prob": item.get("pop", 0.0),  # probability of precipitation
                "rain_3h_mm": item.get("rain", {}).get("3h", 0.0),
                "clouds_pct": item["clouds"]["all"],
                "wind_mps": item["wind"]["speed"],
                "weather": item["weather"][0]["main"],  # Rain/Clear/Clouds/...
            })
        
        df = pd.DataFrame(forecasts)
        return df
    
    def get_daily_summary(self, lat: float, lon: float, days: int = 7) -> pd.DataFrame:
        """
        Tóm tắt theo ngày: mưa nhiều nhất, xác suất mưa cao nhất, nhiệt độ TB
        
        Returns:
            DataFrame với các cột:
            - date: ngày
            - max_rain_prob: xác suất mưa cao nhất trong ngày
            - total_rain_mm: tổng lượng mưa dự kiến (mm)
            - avg_temp_c: nhiệt độ trung bình
            - avg_humidity: độ ẩm trung bình
        """
        df = self.get_forecast(lat, lon)
        if df.empty:
            return df
        
        df["date"] = df["ts"].dt.date
        
        daily = df.groupby("date").agg({
            "rain_prob": "max",
            "rain_3h_mm": "sum",
            "temp_c": "mean",
            "humidity_pct": "mean",
        }).reset_index()
        
        daily.columns = ["date", "max_rain_prob", "total_rain_mm", "avg_temp_c", "avg_humidity"]
        daily = daily.head(days)
        
        return daily

# ===== Demo usage =====
if __name__ == "__main__":
    # Ví dụ: TP. Hồ Chí Minh
    LAT, LON = 10.762622, 106.660172
    
    api = WeatherForecastAPI()
    
    print("=== DỰ BÁO 3H ===")
    forecast_3h = api.get_forecast(LAT, LON)
    print(forecast_3h.head(10))
    
    print("\n=== TÓM TẮT THEO NGÀY ===")
    daily = api.get_daily_summary(LAT, LON, days=5)
    print(daily)
    
    # Lưu cache
    forecast_3h.to_csv("data/weather_forecast_5d.csv", index=False)
    daily.to_csv("data/weather_daily_summary.csv", index=False)
    print("\n✓ Đã lưu vào data/weather_forecast_5d.csv")
```

**Test**: `python src/weather_forecast.py`

---

## 2.3. Cache và lưu trữ dự báo
**File**: `src/weather_cache.py`

```python
"""
Quản lý cache dự báo thời tiết để tránh gọi API quá nhiều
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from weather_forecast import WeatherForecastAPI

DATA_DIR = Path("data")
CACHE_FILE = DATA_DIR / "weather_forecast_cache.csv"
CACHE_HOURS = 3  # refresh mỗi 3 giờ

class WeatherCache:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon
        self.api = WeatherForecastAPI()
    
    def get_forecast(self, force_refresh: bool = False) -> pd.DataFrame:
        """Lấy dự báo, ưu tiên từ cache"""
        now = datetime.utcnow()
        
        # Kiểm tra cache
        if CACHE_FILE.exists() and not force_refresh:
            cache = pd.read_csv(CACHE_FILE, parse_dates=["ts", "cached_at"])
            if not cache.empty:
                last_update = cache["cached_at"].iloc[0]
                if (now - last_update.to_pydatetime().replace(tzinfo=None)) < timedelta(hours=CACHE_HOURS):
                    print(f"✓ Sử dụng cache (cập nhật lúc {last_update})")
                    return cache
        
        # Gọi API mới
        print("⟳ Gọi API OpenWeatherMap...")
        df = self.api.get_forecast(self.lat, self.lon)
        if df.empty:
            return df
        
        df["cached_at"] = now
        df.to_csv(CACHE_FILE, index=False)
        print(f"✓ Đã cache {len(df)} bản ghi dự báo")
        return df

if __name__ == "__main__":
    LAT, LON = 10.762622, 106.660172  # TP.HCM
    cache = WeatherCache(LAT, LON)
    forecast = cache.get_forecast()
    print(forecast.head())
```

---

# GIAI ĐOẠN 3: THUẬT TOÁN LẬP LỊCH TƯỚI THÔNG MINH (3-4 ngày) ⭐⭐⭐

## Mục tiêu:
Xây dựng thuật toán đề xuất lịch tưới 3-7 ngày dựa trên:
- Dự báo thời tiết (API)
- Dự đoán mưa ngắn hạn (30'/60')
- Độ ẩm đất hiện tại
- Nhu cầu nước của cây

---

## 3.1. Mô hình nhu cầu nước của cây
**File**: `src/crop_water_model.py`

```python
"""
Mô hình ước tính nhu cầu nước của cây (đơn giản hóa)
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class CropConfig:
    """Cấu hình cây trồng"""
    name: str
    
    # Độ ẩm đất tối ưu (%)
    optimal_moisture_min: float = 40.0
    optimal_moisture_max: float = 70.0
    
    # Độ ẩm nguy hiểm (%)
    critical_moisture: float = 30.0
    
    # Lượng nước mỗi lần tưới (lít/m²)
    irrigation_amount_lpm2: float = 5.0
    
    # Thời gian tưới (phút)
    irrigation_duration_min: int = 15
    
    # Tần suất tưới tối đa (lần/ngày)
    max_irrigation_per_day: int = 2
    
    # Hệ số thoát hơi (mm/ngày) - phụ thuộc nhiệt độ
    evapotranspiration_base: float = 3.0  # mm/day @ 25°C
    
    def daily_water_need_mm(self, temp_avg: float, humidity_avg: float) -> float:
        """
        Ước tính nhu cầu nước hàng ngày (mm)
        
        Đơn giản: ET = ET_base × k_temp × k_humidity
        """
        # Hệ số nhiệt độ (tăng khi nóng)
        k_temp = 1.0 + (temp_avg - 25) * 0.05
        k_temp = max(0.5, min(2.0, k_temp))
        
        # Hệ số độ ẩm (giảm khi ẩm cao)
        k_humidity = 1.5 - (humidity_avg / 100.0)
        k_humidity = max(0.5, min(1.5, k_humidity))
        
        et = self.evapotranspiration_base * k_temp * k_humidity
        return max(0.5, et)
    
    def irrigation_raises_moisture_by(self) -> float:
        """Lượng nước tưới làm tăng độ ẩm đất bao nhiêu % (ước tính thô)"""
        # Giả sử: 5mm nước = tăng 10% độ ẩm (tùy loại đất)
        return self.irrigation_amount_lpm2 * 2.0

# Ví dụ các loại cây
CROP_PRESETS = {
    "rau_xanh": CropConfig(
        name="Rau xanh",
        optimal_moisture_min=50,
        optimal_moisture_max=75,
        critical_moisture=35,
        irrigation_amount_lpm2=4.0,
        evapotranspiration_base=2.5,
    ),
    "cay_an_trai": CropConfig(
        name="Cây ăn trái",
        optimal_moisture_min=40,
        optimal_moisture_max=70,
        critical_moisture=30,
        irrigation_amount_lpm2=6.0,
        evapotranspiration_base=3.5,
    ),
    "hoa": CropConfig(
        name="Hoa",
        optimal_moisture_min=45,
        optimal_moisture_max=70,
        critical_moisture=32,
        irrigation_amount_lpm2=3.5,
        evapotranspiration_base=2.8,
    ),
}

if __name__ == "__main__":
    crop = CROP_PRESETS["rau_xanh"]
    print(f"Cây: {crop.name}")
    print(f"Độ ẩm tối ưu: {crop.optimal_moisture_min}-{crop.optimal_moisture_max}%")
    print(f"Nhu cầu nước @ 30°C, 60% RH: {crop.daily_water_need_mm(30, 60):.2f} mm/ngày")
```

---

## 3.2. Thuật toán lập lịch tưới
**File**: `src/irrigation_scheduler.py`

```python
"""
Thuật toán lập lịch tưới thông minh 3-7 ngày
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass
from crop_water_model import CropConfig, CROP_PRESETS
from weather_cache import WeatherCache

@dataclass
class IrrigationEvent:
    """Sự kiện tưới"""
    date: str  # YYYY-MM-DD
    time_slot: str  # "morning" | "afternoon" | "evening"
    reason: str
    confidence: float  # 0-1
    predicted_moisture_before: float
    predicted_moisture_after: float

class IrrigationScheduler:
    """Lập lịch tưới thông minh"""
    
    def __init__(self, 
                 crop_type: str,
                 current_moisture: float,
                 lat: float, 
                 lon: float):
        """
        Args:
            crop_type: Loại cây ("rau_xanh", "cay_an_trai", "hoa")
            current_moisture: Độ ẩm đất hiện tại (%)
            lat, lon: Tọa độ để lấy dự báo thời tiết
        """
        self.crop = CROP_PRESETS.get(crop_type, CROP_PRESETS["rau_xanh"])
        self.current_moisture = current_moisture
        self.weather_cache = WeatherCache(lat, lon)
    
    def create_schedule(self, days: int = 7) -> List[IrrigationEvent]:
        """
        Tạo lịch tưới cho N ngày tới
        
        Logic:
        1. Lấy dự báo thời tiết
        2. Tính toán độ ẩm đất dự kiến theo ngày
        3. Quyết định tưới/không tưới dựa trên:
           - Độ ẩm đất < optimal_min → cần tưới
           - Xác suất mưa > 60% → hoãn tưới
           - Lượng mưa dự kiến > 5mm → bỏ qua tưới
           - Độ ẩm < critical → tưới ngay bất kể thời tiết
        """
        # Lấy dự báo
        forecast = self.weather_cache.get_forecast()
        if forecast.empty:
            return []
        
        forecast["date"] = forecast["ts"].dt.date
        daily = forecast.groupby("date").agg({
            "rain_prob": "max",
            "rain_3h_mm": "sum",
            "temp_c": "mean",
            "humidity_pct": "mean",
        }).reset_index()
        daily = daily.head(days)
        
        # Simulate moisture day by day
        schedule: List[IrrigationEvent] = []
        moisture = self.current_moisture
        
        for _, row in daily.iterrows():
            date_str = str(row["date"])
            rain_prob = row["rain_prob"]
            rain_mm = row["rain_3h_mm"]
            temp = row["temp_c"]
            humidity = row["humidity_pct"]
            
            # 1. Tính nhu cầu nước hàng ngày
            et_mm = self.crop.daily_water_need_mm(temp, humidity)
            
            # 2. Giảm độ ẩm do thoát hơi (ước tính: 1mm ET ≈ -2% moisture)
            moisture -= et_mm * 2.0
            
            # 3. Quyết định tưới
            should_irrigate = False
            reason = ""
            confidence = 0.0
            
            # Case 1: Độ ẩm < critical → PHẢI tưới ngay
            if moisture < self.crop.critical_moisture:
                should_irrigate = True
                reason = f"Độ ẩm nguy hiểm ({moisture:.1f}% < {self.crop.critical_moisture}%)"
                confidence = 1.0
            
            # Case 2: Độ ẩm < optimal_min và mưa thấp → nên tưới
            elif moisture < self.crop.optimal_moisture_min:
                if rain_prob < 0.6 and rain_mm < 5.0:
                    should_irrigate = True
                    reason = f"Độ ẩm thấp ({moisture:.1f}%), ít mưa (p={rain_prob:.0%})"
                    confidence = 0.8
                else:
                    reason = f"Hoãn tưới do dự báo mưa (p={rain_prob:.0%}, {rain_mm:.1f}mm)"
                    confidence = 0.5
            
            # Case 3: Độ ẩm OK
            else:
                reason = f"Độ ẩm ổn định ({moisture:.1f}%)"
                confidence = 0.9
            
            # 4. Thực hiện tưới
            if should_irrigate:
                # Tưới vào buổi sáng/chiều (tùy điều kiện)
                time_slot = "morning" if temp < 30 else "evening"
                
                event = IrrigationEvent(
                    date=date_str,
                    time_slot=time_slot,
                    reason=reason,
                    confidence=confidence,
                    predicted_moisture_before=moisture,
                    predicted_moisture_after=moisture + self.crop.irrigation_raises_moisture_by()
                )
                schedule.append(event)
                
                # Cập nhật độ ẩm sau tưới
                moisture += self.crop.irrigation_raises_moisture_by()
            
            # 5. Tăng độ ẩm nếu có mưa
            if rain_mm > 0:
                moisture_gain = min(rain_mm * 2.0, 20.0)  # 1mm rain ≈ +2% moisture (max 20%)
                moisture += moisture_gain
            
            # Giới hạn độ ẩm trong [0, 100]
            moisture = max(0.0, min(100.0, moisture))
        
        return schedule
    
    def print_schedule(self, schedule: List[IrrigationEvent]):
        """In lịch tưới đẹp"""
        print(f"\n{'='*70}")
        print(f"LỊCH TƯỚI CHO {self.crop.name.upper()}")
        print(f"Độ ẩm hiện tại: {self.current_moisture:.1f}%")
        print(f"Độ ẩm tối ưu: {self.crop.optimal_moisture_min}-{self.crop.optimal_moisture_max}%")
        print(f"{'='*70}\n")
        
        if not schedule:
            print("❌ Không cần tưới trong khoảng thời gian này.\n")
            return
        
        for i, event in enumerate(schedule, 1):
            print(f"🌱 Lần {i}: {event.date} ({event.time_slot})")
            print(f"   Lý do: {event.reason}")
            print(f"   Độ ẩm: {event.predicted_moisture_before:.1f}% → {event.predicted_moisture_after:.1f}%")
            print(f"   Độ tin cậy: {event.confidence:.0%}\n")

# ===== Demo =====
if __name__ == "__main__":
    # Giả sử đất hiện tại khô (35%)
    scheduler = IrrigationScheduler(
        crop_type="rau_xanh",
        current_moisture=35.0,
        lat=10.762622,
        lon=106.660172
    )
    
    schedule = scheduler.create_schedule(days=7)
    scheduler.print_schedule(schedule)
    
    # Export JSON
    import json
    schedule_json = [
        {
            "date": e.date,
            "time_slot": e.time_slot,
            "reason": e.reason,
            "confidence": e.confidence,
            "moisture_before": e.predicted_moisture_before,
            "moisture_after": e.predicted_moisture_after,
        }
        for e in schedule
    ]
    
    with open("data/irrigation_schedule.json", "w", encoding="utf-8") as f:
        json.dump(schedule_json, f, ensure_ascii=False, indent=2)
    
    print("✓ Đã lưu vào data/irrigation_schedule.json")
```

**Test**: `python src/irrigation_scheduler.py`

---

## 3.3. Tích hợp với nowcast model
**File**: `src/irrigation_decision_realtime.py`

```python
"""
Quyết định tưới NGAY BÂY GIỜ hay không (sử dụng nowcast 30'/60')
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict

MODEL_DIR = Path("models")

class RealTimeIrrigationDecision:
    """Quyết định tưới real-time dựa vào nowcast"""
    
    def __init__(self):
        # Load models
        self.model_30 = joblib.load(MODEL_DIR / "xgb_nowcast_30min.pkl")
        self.model_60 = joblib.load(MODEL_DIR / "xgb_nowcast_60min.pkl")
    
    def should_irrigate_now(self, 
                           sensor_features: np.ndarray,
                           current_moisture: float,
                           threshold_moisture: float = 40.0) -> Dict:
        """
        Quyết định: tưới ngay bây giờ hay không?
        
        Args:
            sensor_features: Vector feature từ cảm biến (shape: [1, n_features])
            current_moisture: Độ ẩm đất hiện tại (%)
            threshold_moisture: Ngưỡng cần tưới (%)
        
        Returns:
            {
                "should_irrigate": bool,
                "reason": str,
                "rain_prob_30min": float,
                "rain_prob_60min": float,
                "confidence": float,
            }
        """
        # Dự đoán mưa
        p30 = float(self.model_30.predict_proba(sensor_features)[0, 1])
        p60 = float(self.model_60.predict_proba(sensor_features)[0, 1])
        
        # Logic quyết định
        should_irrigate = False
        reason = ""
        confidence = 0.0
        
        # Case 1: Độ ẩm < threshold và ít mưa → TƯỚI
        if current_moisture < threshold_moisture:
            if p30 < 0.3 and p60 < 0.4:
                should_irrigate = True
                reason = f"Độ ẩm thấp ({current_moisture:.1f}%), ít khả năng mưa"
                confidence = 0.9
            elif p30 >= 0.7 or p60 >= 0.7:
                should_irrigate = False
                reason = f"HOÃN tưới: Khả năng mưa cao (30'={p30:.0%}, 60'={p60:.0%})"
                confidence = 0.8
            else:
                should_irrigate = True
                reason = f"Tưới thận trọng (độ ẩm {current_moisture:.1f}%, mưa trung bình)"
                confidence = 0.6
        else:
            should_irrigate = False
            reason = f"Độ ẩm đủ ({current_moisture:.1f}% >= {threshold_moisture}%)"
            confidence = 0.95
        
        return {
            "should_irrigate": should_irrigate,
            "reason": reason,
            "rain_prob_30min": p30,
            "rain_prob_60min": p60,
            "confidence": confidence,
            "current_moisture": current_moisture,
        }

if __name__ == "__main__":
    # Demo
    decision = RealTimeIrrigationDecision()
    
    # Giả lập sensor features (dummy)
    dummy_features = np.random.randn(1, 19).astype("float32")
    
    result = decision.should_irrigate_now(
        sensor_features=dummy_features,
        current_moisture=35.0,
        threshold_moisture=40.0
    )
    
    print("\n=== QUYẾT ĐỊNH TƯỚI REAL-TIME ===")
    print(f"Nên tưới: {'✅ CÓ' if result['should_irrigate'] else '❌ KHÔNG'}")
    print(f"Lý do: {result['reason']}")
    print(f"Độ tin cậy: {result['confidence']:.0%}")
    print(f"Khả năng mưa 30': {result['rain_prob_30min']:.1%}")
    print(f"Khả năng mưa 60': {result['rain_prob_60min']:.1%}")
```

---

# GIAI ĐOẠN 4: TÍCH HỢP VÀO WEB UI (1-2 ngày)

## 4.1. Thêm endpoint lịch tưới vào Flask app
**File update**: `src/app.py`

Thêm các route mới:

```python
@app.route("/schedule")
def schedule_page():
    """Trang lịch tưới 7 ngày"""
    device = request.args.get("device", "esp32-01")
    crop_type = request.args.get("crop", "rau_xanh")
    
    # Lấy độ ẩm hiện tại
    latest = df[df["device_id"] == device].tail(1)
    if latest.empty:
        return "No data", 404
    
    current_moisture = float(latest.iloc[0]["soil_moist_pct"])
    
    # Tạo lịch
    from irrigation_scheduler import IrrigationScheduler
    scheduler = IrrigationScheduler(
        crop_type=crop_type,
        current_moisture=current_moisture,
        lat=10.762622,  # TODO: lấy từ device config
        lon=106.660172
    )
    schedule = scheduler.create_schedule(days=7)
    
    # Render template (tạo template HTML riêng)
    return render_template_string(SCHEDULE_TEMPLATE, 
                                 schedule=schedule,
                                 crop=scheduler.crop,
                                 current_moisture=current_moisture)

@app.route("/decision")
def decision_now():
    """API quyết định tưới ngay bây giờ"""
    device = request.args.get("device", "esp32-01")
    
    latest = df[df["device_id"] == device].tail(1)
    if latest.empty:
        return jsonify({"error": "No data"}), 404
    
    row = latest.iloc[0]
    x = row_to_feature_vector(row)
    current_moisture = float(row["soil_moist_pct"])
    
    from irrigation_decision_realtime import RealTimeIrrigationDecision
    decider = RealTimeIrrigationDecision()
    result = decider.should_irrigate_now(x, current_moisture)
    
    return jsonify(result)
```

---

# 📚 PHẦN PHỤ LỤC: CẤU TRÚC DỰ ÁN SAU KHI BỔ SUNG

```
ai_weather_nowcast/
├── data/
│   ├── sensor_raw_60d.csv
│   ├── labels_rain_30d.csv          # ← MỚI
│   ├── labels_rain_60d_fixed.csv
│   ├── weather_forecast_5d.csv      # ← MỚI (cache API)
│   ├── weather_daily_summary.csv    # ← MỚI
│   └── irrigation_schedule.json     # ← MỚI
│
├── models/
│   ├── xgb_nowcast.pkl              # 60min
│   ├── xgb_nowcast_30min.pkl        # ← MỚI
│   ├── xgb_amount.pkl               # 60min
│   ├── xgb_amount_30min.pkl         # ← MỚI
│   ├── metadata.json
│   ├── metadata_30min.json          # ← MỚI
│   └── metadata_amount.json
│
├── src/
│   ├── app.py                       # Flask UI (CẬP NHẬT)
│   ├── train_xgb_nowcast.py         # 60min (CŨ)
│   ├── train_xgb_nowcast_30min.py   # ← MỚI
│   ├── train_xgb_amount.py
│   ├── train_xgb_amount_30min.py    # ← MỚI
│   ├── wrappers.py
│   ├── weather_forecast.py          # ← MỚI (API OWM)
│   ├── weather_cache.py             # ← MỚI (cache)
│   ├── crop_water_model.py          # ← MỚI (nhu cầu nước)
│   ├── irrigation_scheduler.py      # ← MỚI (lịch 7 ngày)
│   └── irrigation_decision_realtime.py  # ← MỚI (quyết định ngay)
│
├── scripts/
│   └── create_labels_30min.py       # ← MỚI
│
├── .env                             # ← MỚI (chứa OWM_API_KEY)
├── requirements.txt                 # CẬP NHẬT (thêm requests)
├── README.md
└── ROADMAP_ENHANCEMENT.md           # ← TÀI LIỆU NÀY
```

---

# 🎯 TỔNG KẾT ƯU TIÊN

## Priority 1 (NGAY LẬP TỨC - 1-2 ngày):
✅ Train model nowcast 30 phút
✅ Tích hợp API OpenWeatherMap

## Priority 2 (QUAN TRỌNG - 3-4 ngày):
✅ Xây dựng thuật toán lập lịch tưới
✅ Module quyết định tưới real-time

## Priority 3 (BỔ SUNG - 1-2 ngày):
✅ Tích hợp vào Web UI
✅ Test và tối ưu

---

# ⚡ BƯỚC TIẾP THEO

1. **Tạo các thư mục cần thiết:**
   ```bash
   mkdir -p scripts
   ```

2. **Đăng ký API OpenWeatherMap** (10 phút)

3. **Chạy lần lượt các script theo roadmap**

4. **Test từng module trước khi tích hợp**

---

**Hết roadmap. Bạn có muốn tôi bắt đầu triển khai code không?**

