import redis from '../redis/client.js'
import { startTimer } from '../room/timerManager.js'

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

    let result
    try {
        const raw = await redis.placeBidAtomic(
            `room:${roomId}:current`,
            `room:${roomId}:team:${teamId}`,
            `room:${roomId}`,
            teamId
        )
        result = JSON.parse(raw)
    } catch (err) {
        console.error('[onBid] Lua script execution failed:', err)
        return socket.emit('bidError', { message: 'Something went wrong placing your bid. Please try again.' })
    }

    if (!result.success) {
        return socket.emit('bidError', { message: result.error })
    }

    // Bid accepted atomically — now reset the timer for a fresh 30s window.
    try {
        await startTimer(io, roomId)
    } catch (err) {
        console.error(`[onBid] startTimer failed after successful bid in room ${roomId}:`, err)
        io.to(roomId).emit('bidPlaced', {
            teamId,
            newBid: result.newBid,
            timerWarning: 'Timer sync issue — refresh if the countdown looks wrong.'
        })
        return
    }

    io.to(roomId).emit('bidPlaced', {
        teamId,
        newBid: result.newBid
    })
}

export { onBid }