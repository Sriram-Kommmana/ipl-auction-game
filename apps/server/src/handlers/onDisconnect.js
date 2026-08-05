import redis from '../redis/client.js'
import {
    unregisterSocket,
    getPlayerIdBySocket,
    startGraceTimer
} from '../socket/socketRegistry.js'
import { pauseTimer } from '../room/timerManager.js'
import { RESULT_DISPLAY_DURATION } from '../constants.js'

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

    const [room, playerData] = await Promise.all([
        redis.hgetall(`room:${roomId}`),
        redis.hget(`room:${roomId}:players`, playerId)
    ])

    if (!room || Object.keys(room).length === 0) {
        return
    }

    const isManager = room.managerPlayerId === playerId

    if (isManager && (room.status === 'active' || room.status === 'paused')) {
        io.to(roomId).emit('managerDisconnected', {
            gracePeriodSeconds: GRACE_PERIOD_MS / 1000,
            message:            'Manager disconnected. Auction will pause in 15 seconds if they do not reconnect.'
        })

        startGraceTimer(roomId, async () => {
            console.log(`[onDisconnect] Grace period expired for room ${roomId} — auto-pausing`)
            try {
                // A single attempt can land exactly inside the 2-second
                // PROCESSING_EXPIRY window (see timerManager.js's onTimerExpiry
                // — SOLD!/UNSOLD! banner delay before the next player's timer
                // starts). During that window, timerState briefly isn't
                // 'RUNNING', so pauseTimer would correctly no-op — but the
                // OLD code emitted auctionAutoPaused unconditionally anyway,
                // showing "paused" when nothing had actually paused. Now:
                // pauseTimer reports success/failure, and we retry once
                // after that window passes rather than giving up or lying
                // about what happened.
                const attemptPause = async () => {
                    const currentRoom = await redis.hgetall(`room:${roomId}`)
                    if (!currentRoom || currentRoom.status !== 'active') return false
                    return await pauseTimer(io, roomId)
                }

                let paused = await attemptPause()

                if (!paused) {
                    await new Promise((resolve) =>
                        setTimeout(resolve, RESULT_DISPLAY_DURATION + 500)
                    )
                    paused = await attemptPause()
                }

                if (paused) {
                    io.to(roomId).emit('auctionAutoPaused', {
                        message: 'Auction paused — manager did not reconnect in time.'
                    })
                } else {
                    console.log(`[onDisconnect] Grace expired for room ${roomId} but could not pause (room may have ended, or manager reconnected) — skipping notice`)
                }
            } catch (err) {
                console.error(`[onDisconnect] Auto-pause failed for room ${roomId}:`, err)
            }
        }, GRACE_PERIOD_MS)
    }

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