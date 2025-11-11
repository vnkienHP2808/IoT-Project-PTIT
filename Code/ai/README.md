# Dự án: Dự báo Thời tiết Cục bộ (AI Standalone)

## 1) Mục tiêu

Xây dựng mô-đun AI **độc lập** (Python-only) dự báo mưa ngắn hạn (nowcasting 30/60 phút) từ dữ liệu cảm biến tại chỗ: nhiệt độ, độ ẩm, áp suất, ẩm đất, lượng mưa 5’. Kết quả trả về **JSON**. Mô-đun này dùng để thử nghiệm/thẩm định mô hình trước khi tích hợp với server hay frontend.

---

## 2) Cấu trúc thư mục

```
ai-weather-nowcast/
  README.md
  requirements.txt
  data/
    sensor_raw_60d.csv
    labels_rain_60d.csv
    irrigation_events_60d.csv            # (tùy chọn, giúp tạo thêm feature tưới)
    external_weather_60d.csv             # (tùy chọn, mô phỏng API thời tiết)
  artifacts/
    model_xgb_rain60.pkl                 # sinh sau khi train
    feature_schema.json                  # sinh sau khi train (thứ tự feature)
  src/
    train_xgb.py                         # train + lưu model .pkl
    infer_cli.py                         # CLI trả JSON
    app.py                               # FastAPI REST trả JSON
```

> Bạn chỉ cần copy đúng cây thư mục, đặt 2 file dữ liệu bắt buộc vào `data/`, cài `requirements.txt`, chạy train → infer.

---

## 3) Cài đặt nhanh

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
pandas
numpy
scikit-learn
xgboost
fastapi
uvicorn
joblib
requests
```

---

## 4) Huấn luyện (train)

```bash
python src/train_xgb.py
```

Sinh ra:

```
artifacts/model_xgb_rain60.pkl
artifacts/feature_schema.json
```

---

## 5) Suy luận (inference)

### Cách 1 – CLI (in JSON)

```bash
python src/infer_cli.py                      # lấy mốc thời gian mới nhất
python src/infer_cli.py 2025-09-25T09:10:00Z # chỉ định mốc thời gian
```

Ví dụ JSON:

```json
{"ts":"2025-09-25 09:10:00+00:00","device_id":"esp32-01","p_rain_60":0.2371,"model":"model_xgb_rain60.pkl"}
```

### Cách 2 – REST (FastAPI)

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Gọi thử:

```bash
curl "http://localhost:8000/infer?device_id=esp32-01"
```

---

## 6) Tạo đặc trưng (feature) và vì sao

* **Hiện tại**: `temp_c, rh_pct, pressure_hpa, soil_moist_pct, rain_mm_5min`
* **Lags 15’**: `_lag15` cho mỗi biến
* **Rolling 30’**: `_mean30` cho mỗi biến
* **Deltas 15’**: `pressure_delta15, rh_delta15, temp_delta15`
* **Cờ (flags)**: `rain_in_last_15m`
* **Thời gian**: `hour_of_day, day_of_week`
* **(Tuỳ chọn – từ irrigation_events)**: `is_irrigating_now, time_since_last_irrig_min, irrig_duration_last, irrig_total_min_last_3h, irrig_total_min_last_6h`
* **(Tuỳ chọn – từ external weather/OWM)**: `api_rain_mm_60, api_pop_60, wind_mps, cloud_pct, temp_api, rh_api`

Mục tiêu huấn luyện: **`rain_next_60`** – có mưa trong 60 phút tới hay không (0/1).

---

## 7) Thêm dữ liệu OpenWeatherMap (OWM) – gợi ý

* Tạo API key → gọi endpoint **5 day / 3 hour forecast** (`/data/2.5/forecast`)
* Parse JSON → nội suy về mỗi giờ → merge-asof với khung 5 phút của bạn → lưu thành `data/sensor_raw_60d_with_api.csv` rồi thêm các cột API vào `FEATURES` trong `train_xgb.py`.

Mẫu ETL đã mô tả trong cuộc trao đổi trước; có thể dán vào một file riêng như `src/etl_openweather.py` nếu cần.

---

