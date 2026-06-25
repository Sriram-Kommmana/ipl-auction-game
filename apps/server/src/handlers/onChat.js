import redis from '../redis/client.js'
import { v4 as uuidv4 } from 'uuid'

const FOUR_DAYS_IN_SECONDS = 4 * 24 * 60 * 60
const MAX_MESSAGE_LENGTH   = 200
const MAX_CHAT_HISTORY     = 200

const onChat = async (io, socket, data) => {
    const { playerId, text } = data || {}

    if (!playerId) {
        return socket.emit('chatError', { message: 'playerId is required' })
    }

    if (!text || typeof text !== 'string') {
        return socket.emit('chatError', { message: 'Message text is required' })
    }

    const cleanText = text.trim()

    if (cleanText.length === 0) {
        return socket.emit('chatError', { message: 'Message cannot be empty' })
    }

    if (cleanText.length > MAX_MESSAGE_LENGTH) {
        return socket.emit('chatError', { message: `Message cannot exceed ${MAX_MESSAGE_LENGTH} characters` })
    }

    const session = await redis.hgetall(`session:${playerId}`)
    if (!session || !session.roomId) {
        return socket.emit('chatError', { message: 'Session not found. Please rejoin the room.' })
    }

    const { roomId } = session

    const playerData = await redis.hget(`room:${roomId}:players`, playerId)
    if (!playerData) {
        return socket.emit('chatError', { message: 'You are no longer part of this room.' })
    }

    const [nickname] = playerData.split(':')

    const message = {
        messageId: uuidv4(),
        playerId,
        nickname,
        type:      'message',
        text:      cleanText,
        sentAt:    Math.floor(Date.now() / 1000)
    }

    const pipeline = redis.pipeline()

    pipeline.lpush(`room:${roomId}:chat`, JSON.stringify(message))
    pipeline.ltrim(`room:${roomId}:chat`, 0, MAX_CHAT_HISTORY - 1)
    pipeline.expire(`room:${roomId}:chat`, FOUR_DAYS_IN_SECONDS)

    await pipeline.exec()

    io.to(roomId).emit('newChatMessage', message)
}

export { onChat }