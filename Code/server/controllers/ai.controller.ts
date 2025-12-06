import logger from '../utils/log';
import Forecast from '../models/Forecast';
import Schedule from '../models/Schedule';
import fs from 'fs';
import path from 'path';
import mongoose from 'mongoose';
import { io } from '..';
import Event from '../shared/constants/event';

// Định nghĩa thư mục lưu file JSON
// Thư mục này sẽ nằm tại: Code/server/ai_data
const AI_DATA_FOLDER = path.join(__dirname, '../ai_data');

// Đảm bảo thư mục tồn tại, nếu chưa có thì tạo mới
const ensureFolderExists = () => {
  if (!fs.existsSync(AI_DATA_FOLDER)) {
    fs.mkdirSync(AI_DATA_FOLDER, { recursive: true });
  }
};

/**
 * Xử lý dữ liệu dự báo mưa từ AI
 * Topic: ai/forecast/rain
 */
export const handleRainForecast = async (payload: string) => {
  try {
    logger.info('--- Nhận được dữ liệu dự báo mưa (AI) ---');
    
    // Parse JSON
    const data = JSON.parse(payload);
    
    // xuất ra file json
    ensureFolderExists();
    const filePath = path.join(AI_DATA_FOLDER, 'rain_forecast.json');
    
    // Ghi file (ghi đè nội dung cũ bằng nội dung mới nhất)
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), { encoding: 'utf-8' });
    logger.info(`Đã xuất dữ liệu ra file: ${filePath}`);


  } catch (error) {
    logger.error(`Lỗi xử lý topic ai/forecast/rain: ${error}`);
  }
};

/**
 * Xử lý lịch tưới từ AI
 * Topic: ai/schedule/irrigation
 */
export const handleIrrigationSchedule = async (payload: string) => {
  let session: mongoose.ClientSession | null = null;

  try {
    logger.info('Nhận được lịch tưới từ AI');

    const data = JSON.parse(payload);

    if (!data.slots || !Array.isArray(data.slots)) {
      logger.warn('Payload không có mảng slots hợp lệ, bỏ qua xử lý');
      return;
    }

    session = await mongoose.startSession();
    session.startTransaction();

    // Xóa lịch cũ từ hôm nay trở đi
    const cutoffDate = new Date();
    cutoffDate.setHours(0, 0, 0, 0);

    const deleteResult = await Schedule.deleteMany({
      start: { $gte: cutoffDate }
    }, { session });

    logger.info(`Đã xóa ${deleteResult.deletedCount} lịch tưới cũ`);

    // Chuẩn bị dữ liệu lưu DB + dữ liệu gom nhóm để emit về FE
    const schedulesToSave: any[] = [];
    const groupedByDate = new Map<string, any[]>();

    data.slots.forEach((slot: any) => {
      const start = new Date(slot.start_ts);
      const end = new Date(slot.end_ts);

      if (isNaN(start.getTime()) || isNaN(end.getTime())) {
        logger.warn(`Slot thời gian không hợp lệ, bỏ qua: ${JSON.stringify(slot)}`);
        return;
      }

      const dateKey = slot.date; // YYYY-MM-DD
      const timeStart = formatTime(start);
      const timeEnd = formatTime(end);

      // Dữ liệu để lưu vào DB
      schedulesToSave.push({
        start,
        end,
        durationMin: slot.duration_min,
        date: dateKey,
        note: `Tưới ${slot.duration_min} phút [${timeStart} to ${timeEnd}]`,
      });

      // Gom nhóm theo ngày để gửi FE
      if (!groupedByDate.has(dateKey)) {
        groupedByDate.set(dateKey, []);
      }

      groupedByDate.get(dateKey)!.push({
        start: timeStart,        // "07:00"
        end: timeEnd,            // "07:20"
        durationMin: slot.duration_min,
      });
    });

    // Chuyển Map → Array để dễ xử lý ở FE
    const groupedDataForFE = Array.from(groupedByDate.entries())
      .map(([date, slots]) => ({
        date, // "2025-12-09"
        slots: slots.sort((a, b) => a.start.localeCompare(b.start)) // sắp xếp theo giờ tăng dần
      }))
      .sort((a, b) => a.date.localeCompare(b.date)); // sắp xếp ngày tăng dần

    // Lưu vào DB
    if (schedulesToSave.length > 0) {
      await Schedule.insertMany(schedulesToSave, { session });
      logger.info(`Đã lưu ${schedulesToSave.length} lịch tưới vào DB`);
    }

    await session.commitTransaction();
    logger.info('Cập nhật lịch tưới thành công');

    // Emit về Frontend
    if (io) {
      const emitData = {
        data: groupedDataForFE
      };

      io.emit(Event.IRRIGATION_SCHEDULE_UPDATE, emitData);
      logger.info(`Đã emit lịch tưới mới tới client (${groupedDataForFE.length} ngày)`);
    }

  } catch (error) {
    logger.error(`Lỗi xử lý lịch tưới AI: ${error}`);
    if (session) await session.abortTransaction();
  } finally {
    if (session) session.endSession();
  }
};

/**
 * Format Date → chuỗi HH:mm
 */
const formatTime = (date: Date): string => {
  const pad = (n: number) => n.toString().padStart(2, '0');
  return `${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}`;
};