import logger from '../utils/log';
import SensorData from '../models/SensorData';
import mqttClient from '../config/mqtt.config';
import { Server } from 'socket.io';
import { io } from '..';
import Topic from '../shared/constants/topic';
import Event from '../shared/constants/event';
import HTTPStatus from '../shared/constants/httpStatus';
import { AuthRequest } from '../shared/types/util.type';
import { Response } from 'express';
import { now } from 'mongoose';

/**
 * Hàm làm tròn số 2 số sau thập phân
 * @params num
 * @return {số thập phân sau khi xử lý}
 */
const roundToTwo = (num: number): number => {
  if (typeof num !== 'number') return num;
  return parseFloat(num.toFixed(2));
};

/**
 * Hàm chuẩn hóa Date thành "dd/MM/yyyy"
 * @param date
 * @returns {định dạng ngày sau khi xử lý}
 */
const formatDate = (date: string): string => {
  const newdate = new Date(date);

  // Lấy ngày, tháng, năm
  const day = newdate.getDate().toString().padStart(2, '0');
  const month = (newdate.getMonth() + 1).toString().padStart(2, '0'); // getMonth() bắt đầu từ 0
  const year = newdate.getFullYear();

  // Lấy giờ, phút, giây
  const hours = newdate.getHours().toString().padStart(2, '0');
  const minutes = newdate.getMinutes().toString().padStart(2, '0');
  const seconds = newdate.getSeconds().toString().padStart(2, '0');

  // Kết hợp lại
  return `${day}/${month}/${year} ${hours}:${minutes}:${seconds}`;
};

/**
 * Xử lý tin nhắn từ topic 'sensor/data/push'
 */
const handleSensorData = async (payload: string) => {
  try {
    const data = JSON.parse(payload);
    
    // biến để gửi thẳng lên client qua socket
    const emitData = {
      temperature: roundToTwo(data.temperature),
      humidity: roundToTwo(data.humidity),
      pressureHpa: roundToTwo(data.pressure_hpa),
      soilMoisture: roundToTwo(data.soilMoisture),
      timestamp: formatDate(data.timestamp)
    };

    // biến model để lưu vào db
    const newData = new SensorData(emitData)

    const savedData = await newData.save();
    logger.info(`Đã lưu sensor data (MQTT): ${savedData._id}`);

    // phát sự kiện tới client, bên client lắng nghe để lấy dữ liệu
    if(io){
      io.emit(Event.SENSOR_DATA_PUSH, emitData)
      logger.info(`Đã emit "sensor/data/push" tới client`);
      logger.info(JSON.stringify(emitData));
    }

  } catch (error) {
    logger.error(`Lỗi xử lý tin nhắn 'sensor/data/push': ${error}`);
  }
};

const getSensorData = async (req: AuthRequest, res: Response) => {
  try {
    // === Lấy thời gian 24 giờ trước (1 ngày) ===
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
    const now = new Date();

    // Format thành chuỗi đúng định dạng đang lưu trong DB
    const formatToDDMMYYYY = (date: Date): string => {
      const day = String(date.getDate()).padStart(2, '0');
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const year = date.getFullYear();
      const hours = String(date.getHours()).padStart(2, '0');
      const minutes = String(date.getMinutes()).padStart(2, '0');
      const seconds = String(date.getSeconds()).padStart(2, '0');
      return `${day}/${month}/${year} ${hours}:${minutes}:${seconds}`;
    };

    const oneDayAgoStr = formatToDDMMYYYY(oneDayAgo);
    const nowStr = formatToDDMMYYYY(now);

    // Dùng aggregation 
    const rawData = await SensorData.aggregate([
      {
        $match: {
          $expr: {
            $and: [
              // timestamp >= 24h trước
              {
                $gte: [
                  {
                    $dateFromString: {
                      dateString: '$timestamp',
                      format: '%d/%m/%Y %H:%M:%S',
                      onError: new Date(0),
                      onNull: new Date(0),
                    },
                  },
                  { $dateFromString: { dateString: oneDayAgoStr, format: '%d/%m/%Y %H:%M:%S' } },
                ],
              },
              // timestamp <= hiện tại
              {
                $lte: [
                  {
                    $dateFromString: {
                      dateString: '$timestamp',
                      format: '%d/%m/%Y %H:%M:%S',
                      onError: new Date(0),
                      onNull: new Date(0),
                    },
                  },
                  { $dateFromString: { dateString: nowStr, format: '%d/%m/%Y %H:%M:%S' } },
                ],
              },
            ],
          },
        },
      },
      {
        $project: {
          temperature: 1,
          pressureHpa: 1,
          soilMoisture: 1,
          timestamp: 1,
        },
      },
    ]);

    if (rawData.length === 0) {
      return res.status(HTTPStatus.OK).json({
        status: HTTPStatus.OK,
        message: 'Không có dữ liệu trong 24 giờ gần nhất',
        data: {
          temperatureArr: [],
          pressureArr: [],
          soilMoistureArr: [],
        },
      });
    }

    // Tạo mảng dữ liệu 
    const temperatureArr = rawData.map(item => ({
      time: item.timestamp,
      value: Number(roundToTwo(item.temperature)),
    }));


    const pressureArr = rawData.map(item => ({
      time: item.timestamp,
      value: Number(roundToTwo(item.pressureHpa)),
    }));

    const soilMoistureArr = rawData.map(item => ({
      time: item.timestamp,
      value: Number(roundToTwo(item.soilMoisture)),
    }));

    return res.status(HTTPStatus.OK).json({
      status: HTTPStatus.OK,
      message: 'Lấy dữ liệu cảm biến 24 giờ gần nhất thành công',
      data: {
        temperatureArr,
        pressureArr,
        soilMoistureArr,
      },
    });
  } catch (error: any) {
    logger.error(`Lỗi khi lấy dữ liệu cảm biến 24h: ${error.message || error}`);
    return res.status(HTTPStatus.INTERNAL_SERVER_ERROR).json({
      status: HTTPStatus.INTERNAL_SERVER_ERROR,
      message: 'Lỗi server',
    });
  }
};

export {handleSensorData, getSensorData}