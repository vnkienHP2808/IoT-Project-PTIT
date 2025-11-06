import 'dotenv/config'
import express from 'express'
import { clearDataSensorData, connectDB } from './config/db.config'
import './models/SensorData'
import './models/Forecast'
import './models/Schedule'
import cors from 'cors'

const app = express()
const PORT = process.env.PORT || 5000
app.use(
  cors({
    origin: process.env.ALLOW_ORIGIN
  })
)

app.use(express.json())
const startServer = async () => {
  await connectDB()
  // await clearDataSensorData()
  app.listen(PORT, () => console.log(`Server đang chạy trên cổng ${PORT}`))
}

startServer()
