import redis from '../redis/client.js'
import { resumeTimer } from '../room/timerManager.js'

const onResume = async (io, socket, data) => {
    const { playerId } = data || {}

    if (!playerId) {
        return socket.emit('resumeError', { message: 'playerId is required' })
    }

    const session = await redis.hgetall(`session:${playerId}`)
    if (!session || !session.roomId) {
        return socket.emit('resumeError', { message: 'Session not found. Please rejoin the room.' })
    }

    const { roomId } = session

    const isMember = await redis.hexists(`room:${roomId}:players`, playerId)
    if (!isMember) {
        return socket.emit('resumeError', { message: 'You are no longer part of this room.' })
    }

    const room = await redis.hgetall(`room:${roomId}`)
    if (Object.keys(room).length === 0) {
        return socket.emit('resumeError', { message: 'Room not found or has expired.' })
    }

    if (room.managerPlayerId !== playerId) {
        return socket.emit('resumeError', { message: 'Only the manager can resume the auction.' })
    }

    if (room.status !== 'paused') {
        return socket.emit('resumeError', { message: 'Auction is not currently paused.' })
    }

    await resumeTimer(io, roomId)
}

export { onResume }