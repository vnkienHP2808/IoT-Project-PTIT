import { Request, Response } from 'express'
import User from "../models/User"
import HTTPStatus from "../shared/constants/httpStatus"
import logger from "../utils/log"
import * as jwt from 'jsonwebtoken'


const login = async (req: Request, res: Response) => {
  const { username, password } = req.body
  logger.info(`User [${username}] đăng nhập vào hệ thống`)
  if (!username || !password) {
    return res.status(HTTPStatus.BAD_REQUEST).json({
      status: HTTPStatus.BAD_REQUEST,
      message: 'Vui lòng cung cấp đầy đủ tên đăng nhập và mật khẩu.'
    })
  }
  try {
    const user = await User.findOne({ username: username })
    if (!user) {
      return res.status(HTTPStatus.UNAUTHORIZED).json({
        status: HTTPStatus.UNAUTHORIZED,
        message: 'Tên đăng nhập hoặc mật khẩu không đúng.'
      })
    }
    const isPasswordValid = password === user.password
    if (!isPasswordValid) {
      return res.status(HTTPStatus.UNAUTHORIZED).json({
        status: HTTPStatus.UNAUTHORIZED,
        message: 'Tên đăng nhập hoặc mật khẩu không đúng.'
      })
    }

    // đầu tiên định cho chung mà đọc api design bảo chỉ 3 cái này nên tách ra nhé
    const Payload = { 
      id: user.id,
      username: user.username,
      role: user.role
    }
    const access_token = jwt.sign(Payload, process.env.JWT_SECRET as string, {
      expiresIn: '1d'
    })

    const userResponseData = {
      id: user.id,
      username: user.username,
      fullname: user.fullname,
      address: user.address,
      phoneNumber: user.phoneNumber,
      role: user.role
    }
    return res.status(HTTPStatus.OK).json({
      status: HTTPStatus.OK,
      message: 'Đăng nhập thành công',
      data: {
        access_token: access_token,
        user_info: Payload
      }
    })
  } catch (error) {
    console.error('Lỗi đăng nhập:', error)
    return res.status(HTTPStatus.INTERNAL_SERVER_ERROR).json({
      status: HTTPStatus.INTERNAL_SERVER_ERROR,
      message: 'Đã xảy ra lỗi hệ thống.'
    })
  }
}

export { login }