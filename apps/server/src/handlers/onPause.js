import redis from '../redis/client.js'
import { pauseTimer } from '../room/timerManager.js'

const onPause = async (io, socket, data) => {
    const { playerId } = data || {}

    if (!playerId) {
        return socket.emit('pauseError', { message: 'playerId is required' })
    }

    const session = await redis.hgetall(`session:${playerId}`)
    if (!session || !session.roomId) {
        return socket.emit('pauseError', { message: 'Session not found. Please rejoin the room.' })
    }

    const { roomId } = session

    const isMember = await redis.hexists(`room:${roomId}:players`, playerId)
    if (!isMember) {
        return socket.emit('pauseError', { message: 'You are no longer part of this room.' })
    }

    const room = await redis.hgetall(`room:${roomId}`)
    if (Object.keys(room).length === 0) {
        return socket.emit('pauseError', { message: 'Room not found or has expired.' })
    }

    if (room.managerPlayerId !== playerId) {
        return socket.emit('pauseError', { message: 'Only the manager can pause the auction.' })
    }

    if (room.status !== 'active') {
        return socket.emit('pauseError', { message: 'Auction is not currently active.' })
    }

    await pauseTimer(io, roomId)
}

export { onPause }