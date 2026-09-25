import redis from '../redis/client.js'
import { placeBid } from '../services/placeBid.js'

// Socket entry point for a human bid: resolve who is bidding, then hand off
// to the shared placeBid service (the same one server-side bots use).
const onBid = async (io, socket, data) => {
    const { playerId } = data || {}

    if (!playerId) {
        return socket.emit('bidError', { message: 'playerId is required' })
    }

    const session = await redis.hgetall(`session:${playerId}`)
    if (!session || !session.roomId) {
        return socket.emit('bidError', { message: 'Session not found. Please rejoin the room.' })
    }

    const { roomId } = session

    const playerData = await redis.hget(`room:${roomId}:players`, playerId)
    if (!playerData) {
        return socket.emit('bidError', { message: 'You are no longer part of this room.' })
    }

    const teamId = playerData.split(':')[2]
    if (!teamId) {
        return socket.emit('bidError', { message: 'You must select a team before bidding.' })
    }

    const result = await placeBid(io, { roomId, teamId })
    if (!result.success) {
        return socket.emit('bidError', { message: result.error })
    }
}

export { onBid }
