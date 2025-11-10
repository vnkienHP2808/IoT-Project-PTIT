import express from 'express'
import { getCountDevice, getListUser, login } from '../controllers/user.controller'
import { authenticateToken } from '../middlewares/user.middleware'

export const userRouter = express.Router()

userRouter.post('/login', login)

userRouter.use(authenticateToken)
userRouter.get('/list', getListUser)
userRouter.get('/count-device', getCountDevice)