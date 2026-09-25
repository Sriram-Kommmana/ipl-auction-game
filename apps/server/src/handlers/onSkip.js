import redis from '../redis/client.js'
import { v4 as uuidv4 } from 'uuid'
import { clearExistingTimer, startTimer } from '../room/timerManager.js'
import { advanceAuction } from '../room/auctionProgression.js'
import { emitToRoom } from '../services/roomEvents.js'
import { FOUR_DAYS_IN_SECONDS } from '../constants.js'

const onSkip = async (io, socket, data) => {
    const { playerId } = data || {}

    if (!playerId) {
        return socket.emit('skipError', { message: 'playerId is required' })
    }

    const session = await redis.hgetall(`session:${playerId}`)
    if (!session || !session.roomId) {
        return socket.emit('skipError', { message: 'Session not found. Please rejoin the room.' })
    }

    const { roomId } = session

    const isMember = await redis.hexists(`room:${roomId}:players`, playerId)
    if (!isMember) {
        return socket.emit('skipError', { message: 'You are no longer part of this room.' })
    }

    const [room, current] = await Promise.all([
        redis.hgetall(`room:${roomId}`),
        redis.hgetall(`room:${roomId}:current`)
    ])

    if (!room || Object.keys(room).length === 0) {
        return socket.emit('skipError', { message: 'Room not found or has expired.' })
    }

    if (room.managerPlayerId !== playerId) {
        return socket.emit('skipError', { message: 'Only the manager can skip a player.' })
    }

    if (room.status !== 'active' && room.status !== 'paused') {
        return socket.emit('skipError', { message: 'Auction is not currently running.' })
    }

    if (!current || Object.keys(current).length === 0) {
        return socket.emit('skipError', { message: 'No player is currently up for auction.' })
    }

    // Explicit timerState check before attempting the lock —
    // gives a clear error message if called during PROCESSING_EXPIRY or ENDED
    // rather than the generic "transition in progress" message from the lock
    if (current.timerState !== 'RUNNING' && current.timerState !== 'PAUSED') {
        return socket.emit('skipError', {
            message: 'Cannot skip right now. Player transition is already in progress.'
        })
    }

    // Clear any existing countdown first — prevents race with timer firing
    clearExistingTimer(roomId)

    // Atomically acquire the lock — only one of onSkip/onTimerExpiry can proceed.
    // Returns 0 if onTimerExpiry already grabbed it in the same instant.
    // Any timer ('') and paused lots allowed ('') — a skip can end a paused lot.
    const acquired = await redis.acquireExpiryLock(`room:${roomId}:current`, '', '')
    if (!acquired) {
        return socket.emit('skipError', {
            message: 'Player transition already in progress. Please wait.'
        })
    }

    const { iplPlayerId, playerName } = current
    const currentPlayerIndex = Number(room.currentPlayerIndex)
    const auctionPhase       = room.auctionPhase
    const poolLength         = await redis.llen(`room:${roomId}:pool`)

    // soldAt computed once, reused everywhere — same single-source-of-truth
    // pattern as timerManager.js, so the frontend never has to generate its
    // own timestamp (which would drift from the server's clock).
    const soldAt = Math.floor(Date.now() / 1000)

    // history records status as "skipped" (distinct from "unsold")
    // pool:unsold receives the player regardless — both go to re-auction
    const historyEntry = JSON.stringify({
        iplPlayerId,
        playerName,
        soldTo:  null,
        soldFor: null,
        status:  'skipped',
        soldAt
    })

    const chatObj = {
        messageId: uuidv4(),
        playerId:  'system',
        nickname:  'Auction',
        type:      'broadcast',
        text:      `${playerName} was skipped`,
        sentAt:    soldAt
    }
    const chatEntry = JSON.stringify(chatObj)

    try {
        const pipeline = redis.pipeline()
        pipeline.hset(`room:${roomId}:pool:status`, { [iplPlayerId]: 'unsold' })
        pipeline.rpush(`room:${roomId}:pool:unsold`, iplPlayerId)
        pipeline.expire(`room:${roomId}:pool:unsold`, FOUR_DAYS_IN_SECONDS)
        pipeline.rpush(`room:${roomId}:history`, historyEntry)
        pipeline.expire(`room:${roomId}:history`, FOUR_DAYS_IN_SECONDS)
        pipeline.lpush(`room:${roomId}:chat`, chatEntry)
        pipeline.ltrim(`room:${roomId}:chat`, 0, 199)
        pipeline.expire(`room:${roomId}:chat`, FOUR_DAYS_IN_SECONDS)
        // Ensure room is active even if it was paused when manager pressed skip
        pipeline.hset(`room:${roomId}`, { status: 'active' })
        await pipeline.exec()

        emitToRoom(io, roomId, 'playerSkipped', {
            iplPlayerId,
            playerName,
            soldAt
        })
        emitToRoom(io, roomId, 'newChatMessage', chatObj)

        await advanceAuction(io, roomId, currentPlayerIndex, auctionPhase, poolLength, startTimer)

    } catch (err) {
        console.error(`[onSkip] Failed for room ${roomId}:`, err)
        socket.emit('skipError', {
            message: 'Something went wrong while skipping. Please try again.'
        })
    }
}

export { onSkip }