Không, nó không "tự động" . Bạn, với tư cách là người lập trình, phải **lập trình cho cảm biến (ESP32)** làm 2 việc:
1.  Gửi tin nhắn "Tôi Online" khi nó vừa kết nối.
2.  Thiết lập một "Di chúc" để Broker tự động báo "Tôi Offline" khi nó bị ngắt kết nối đột ngột.

---

## Luồng hoạt động trong thực tế (Cảm biến thật)

Đây là chính xác những gì xảy ra.

### 1. Khi Cảm biến Khởi động (Lên Online)

Khi bạn cắm điện cho ESP32 (cảm biến thật):

1.  **Thiết lập "Di chúc" (LWT - Last Will and Testament):**
    * *Trước khi* nhấn "Connect", code trên ESP32 sẽ chuẩn bị một "Di chúc".
    * Nó nói với thư viện MQTT: "Nếu tôi kết nối xong mà bị ngắt kết nối đột ngột (rớt mạng, mất điện) mà không kịp 'chào tạm biệt', thì hãy thay tôi Gửi (Publish) tin nhắn này:"
        * **Topic Di chúc:** `devices/status/esp32-01`
        * **Payload Di chúc (plain text):** `0`
2.  **Kết nối (Connect):**
    * ESP32 thực hiện kết nối đến Broker (HiveMQ) kèm theo "Di chúc" đã thiết lập.
3.  **Gửi tin nhắn "Online" (Birth Message):**
    * *Ngay sau khi* kết nối thành công, dòng code tiếp theo trong ESP32 sẽ là một lệnh **Gửi (Publish)**.
    * **Topic:** `devices/status/esp32-01` (đúng topic trong Di chúc).
    * **Payload (plain text):** `1`
    * **(Quan trọng) Retain Flag (Cờ giữ lại):** `true`.
4.  **Phản ứng của Server:**
    * Server của bạn (đang lắng nghe `devices/status/+`) nhận được tin nhắn (Topic: `devices/status/esp32-01`, Payload: `1`).
    * Hàm `handleDeviceStatus` được gọi.
    * `deviceId` là `"esp32-01"` và `payload` là `"1"`.
    * Server thêm `"esp32-01"` vào danh sách `connectedDevices`. API `getCountDevice` lúc này sẽ trả về `1`.

*(**Tại sao Retain=true?** Vì nếu Server của bạn bị khởi động lại, nó sẽ subscribe lại và *ngay lập tức* nhận được tin nhắn `1` này (do Broker đã "giữ lại"), giúp nó biết `esp32-01` vẫn đang online mà không cần chờ ESP32 gửi lại).*

---

### 2. Khi Cảm biến Ngắt kết nối (Bị Offline)

Đây là phần hay nhất, có 2 trường hợp:

#### Trường hợp A: Ngắt kết nối ĐỘT NGỘT (Rớt mạng, mất điện)

1.  ESP32 bị rút điện. Nó không kịp gửi bất cứ thứ gì.
2.  Broker (HiveMQ) chờ một lúc (vài giây đến vài phút, tùy cấu hình) mà không thấy tín hiệu "keep-alive" từ ESP32.
3.  Broker xác định "ESP32 đã chết".
4.  Broker **THỰC THI DI CHÚC (LWT):** Broker *tự động* Gửi (Publish) tin nhắn mà ESP32 đã đăng ký ở bước 1.
    * **Topic:** `devices/status/esp32-01`
    * **Payload (plain text):** `0`
5.  **Phản ứng của Server:**
    * Server của bạn (đang lắng nghe `devices/status/+`) nhận được tin nhắn (Topic: `devices/status/esp32-01`, Payload: `0`).
    * Hàm `handleDeviceStatus` được gọi.
    * `payload === '1'` là **SAI**.
    * Khối `else` được chạy. Server **xóa** `"esp32-01"` khỏi danh sách `connectedDevices`. API `getCountDevice` giờ trả về `0`.

#### Trường hợp B: Ngắt kết nối CHỦ ĐỘNG (Ví dụ: Cảm biến đi ngủ - Deep Sleep)

1.  ESP32 biết nó sắp đi ngủ (hoặc tắt) một cách an toàn.
2.  *Trước khi* ngắt kết nối, code của ESP32 sẽ chủ động **Gửi (Publish)** một tin nhắn "chào tạm biệt":
    * **Topic:** `devices/status/esp32-01`
    * **Payload (plain text):** `0`
3.  Server nhận được tin `0` này và xóa thiết bị khỏi danh sách (giống như trên).
4.  ESP32 sau đó ngắt kết nối (disconnect) một cách bình thường. Broker thấy đây là ngắt kết nối chủ động nên sẽ **KHÔNG** thực thi "Di chúc".