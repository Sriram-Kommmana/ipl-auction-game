import express from 'express';
import {createRoom} from './createRoom.js'
import {joinRoom} from './joinRoom.js'
import { getRoomState } from './getRoomState.js'

const router = express.Router();

router.post('/create', createRoom)
router.post('/join', joinRoom)
router.get('/:roomId', getRoomState)

export default router;