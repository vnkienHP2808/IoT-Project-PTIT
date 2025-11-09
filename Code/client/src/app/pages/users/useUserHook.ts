import clientService from '@/services/client.service'
import type { User } from '@/shared/types/auth.type'
import { HTTP_STATUS } from '@/shared/types/http.type'
import { useEffect, useState } from 'react'

const useUserHook = () => {
  const [listUser, setListUser] = useState<User[]>([])
  const fetchUser = async () => {
    try {
      const response = await clientService.getListUser()
      console.log('>>>>response', response)
      if (response.status === HTTP_STATUS.OK) {
        const users = response.data.data
        setListUser(users)
      }
    } catch (e) {
      console.log(e)
    }
  }
  useEffect(() => {
    fetchUser()
  }, [])
  return { listUser }
}
export default useUserHook
