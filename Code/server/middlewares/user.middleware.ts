import { Response, NextFunction, Request } from 'express'
import * as jwt from 'jsonwebtoken'
import HTTPStatus from '../shared/constants/httpStatus'


const JWT_SECRET = process.env.JWT_SECRET || 'fallback_secret_key'

export const authenticateToken = (req: Request, res: Response, next: NextFunction) => {
  const authHeader = req.headers['authorization']
  const token = authHeader && authHeader.split(' ')[1]
  if (token == null) {
    return res.status(HTTPStatus.UNAUTHORIZED).json({
      status: HTTPStatus.UNAUTHORIZED,
      message: 'Yêu cầu Access Token. Truy cập bị từ chối.'
    })
  }

  jwt.verify(token, JWT_SECRET, (err) => {
    if (err) {
      return res.status(HTTPStatus.UNAUTHORIZED).json({
        status: HTTPStatus.UNAUTHORIZED,
        message: 'Access Token không hợp lệ hoặc đã hết hạn.',
        data: null
      })
    }
    next()
  })
}