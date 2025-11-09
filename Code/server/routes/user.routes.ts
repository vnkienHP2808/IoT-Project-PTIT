import express from 'express'
import { login } from '../controllers/user.controller'

export const userRouter = express.Router()

userRouter.post('/login', login)
