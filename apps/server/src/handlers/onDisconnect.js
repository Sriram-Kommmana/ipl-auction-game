import redis from '../redis/client.js'
import {
    unregisterSocket,
    getPlayerIdBySocket,
    startGraceTimer
} from '../socket/socketRegistry.js'
import { pauseTimer } from '../room/timerManager.js'

const GRACE_PERIOD_MS = 15 * 1000

const onDisconnect = async (io, socket) => {
    const playerId = getPlayerIdBySocket(socket.id)

    if (!playerId) {
        return
    }

    unregisterSocket(socket.id)

    const session = await redis.hgetall(`session:${playerId}`)
    if (!session || !session.roomId) {
        return
    }

    const { roomId } = session

    // Check room status and manager early — before other Redis writes
    // so we can start the grace timer as soon as possible
    const [room, playerData] = await Promise.all([
        redis.hgetall(`room:${roomId}`),
        redis.hget(`room:${roomId}:players`, playerId)
    ])

    if (!room || Object.keys(room).length === 0) {
        return
    }

    const isManager = room.managerPlayerId === playerId

    // Start grace timer IMMEDIATELY after identifying manager —
    // before any further Redis writes so onReconnect can see it
    if (isManager && (room.status === 'active' || room.status === 'paused')) {
        io.to(roomId).emit('managerDisconnected', {
            gracePeriodSeconds: GRACE_PERIOD_MS / 1000,
            message:            'Manager disconnected. Auction will pause in 15 seconds if they do not reconnect.'
        })

        startGraceTimer(roomId, async () => {
            console.log(`[onDisconnect] Grace period expired for room ${roomId} — auto-pausing`)
            try {
                const currentRoom = await redis.hgetall(`room:${roomId}`)
                if (currentRoom && currentRoom.status === 'active') {
                    await pauseTimer(io, roomId)
                    io.to(roomId).emit('auctionAutoPaused', {
                        message: 'Auction paused — manager did not reconnect in time.'
                    })
                }
            } catch (err) {
                console.error(`[onDisconnect] Auto-pause failed for room ${roomId}:`, err)
            }
        }, GRACE_PERIOD_MS)
    }

    // Now do the remaining Redis writes — presence update happens
    // after grace timer is already running, not before
    if (playerData) {
        const parts = playerData.split(':')
        parts[4] = 'offline'
        await redis.hset(`room:${roomId}:players`, { [playerId]: parts.join(':') })
    }

    socket.to(roomId).emit('playerOffline', {
        playerId,
        nickname: playerData ? playerData.split(':')[0] : ''
    })
}

export { onDisconnect }