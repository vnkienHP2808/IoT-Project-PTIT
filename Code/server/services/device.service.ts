import logger from '../utils/log';

// Danh sách (Set) này sẽ lưu ID của các thiết bị đang online
// Nó chỉ tồn tại trong bộ nhớ (memory) của server
const connectedDevices = new Set<string>();

/**
 * Hàm xử lý tin nhắn trạng thái từ MQTT
 */
export const handleDeviceStatus = (topic: string, payload: string) => {
  try {
    // Tách ID thiết bị từ topic
    // ví dụ: 'devices/status/esp32-01' -> 'esp32-01'
    const deviceId = topic.split('/')[2];
    if (!deviceId) return;

    if (payload === '1' || payload.toLowerCase() === 'online') {
      // Thêm thiết bị vào danh sách
      if (!connectedDevices.has(deviceId)) {
        connectedDevices.add(deviceId);
        logger.info(`Thiết bị [${deviceId}] đã kết nối (online).`);
      }
    } else {
      // Xóa thiết bị khỏi danh sách (payload là '0', 'offline' hoặc rỗng)
      if (connectedDevices.has(deviceId)) {
        connectedDevices.delete(deviceId);
        logger.info(`Thiết bị [${deviceId}] đã ngắt kết nối (offline).`);
      }
    }
    
    logger.info(`Tổng số thiết bị đang kết nối: ${getDeviceCount()}`);

  } catch (error) {
    logger.error('Lỗi xử lý trạng thái thiết bị:', error);
  }
};

/**
 * Hàm lấy số lượng thiết bị đang kết nối
 */
export const getDeviceCount = (): number => {
  return connectedDevices.size;
};