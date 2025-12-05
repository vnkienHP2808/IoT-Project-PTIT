import { useEffect, useState } from 'react'
import { humidityData, pressureData, temperatureData } from '../../dummy'

export type DataType = {
  time: string
  value: number
}

const useChartHook = () => {
  const [temperatureArr, setTemperatureArr] = useState<DataType[]>([])
  const [pressureArr, setPressureArr] = useState<DataType[]>([])
  const [humidityDataArr, setHumidityArr] = useState<DataType[]>([])

  useEffect(() => {
    setTemperatureArr([...temperatureData])
    setPressureArr([...pressureData])
    setHumidityArr([...humidityData])
  }, [])

  return { temperatureArr, pressureArr, humidityDataArr }
}
export default useChartHook
