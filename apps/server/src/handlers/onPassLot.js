import redis from '../redis/client.js'
import { humanPass } from '../bots/botManager.js'

// Solo only: the human is done with the lot on the block. Once no bot will
// bid again either, the lot closes straight away instead of waiting out the
// countdown (see bots/botManager.js).
const onPassLot = async (io, socket, data) => {
    const { playerId } = data || {}
    if (!playerId) {
        return socket.emit('passError', { message: 'playerId is required' })
    }

    const session = await redis.hgetall(`session:${playerId}`)
    if (!session || !session.roomId) {
        return socket.emit('passError', { message: 'Session not found. Please rejoin the room.' })
    }

    const result = await humanPass(session.roomId, playerId)
    if (!result.ok) {
        return socket.emit('passError', { message: result.error })
    }
    socket.emit('lotPassed', { playerId })
}

export { onPassLot }
