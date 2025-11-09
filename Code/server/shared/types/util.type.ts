import { Request } from 'express'
import { UserRole } from '../../models/User';

export interface AuthenticatedRequest extends Request {
  user?: { id: number; username: string; role: UserRole }
}