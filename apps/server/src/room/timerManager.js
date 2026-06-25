import redis from '../redis/client.js'

const TIMER_DURATION = 30

const activeTimers = new Map()

const clearExistingTimer = (roomId) => {
    if (activeTimers.has(roomId)) {
        clearTimeout(activeTimers.get(roomId))
        activeTimers.delete(roomId)
    }
}

const onTimerExpiry = async (io, roomId) => {
    console.log(`[TimerManager STUB] onTimerExpiry fired for room ${roomId}`)
}

const startTimer = async (io, roomId) => {
    const now         = Math.floor(Date.now() / 1000)
    const timerEndsAt = now + TIMER_DURATION

    await redis.hset(`room:${roomId}:current`, {
        timerState:          'RUNNING',
        timerEndsAt:         String(timerEndsAt),
        pausedTimeRemaining: ''
    })

    clearExistingTimer(roomId)

    const timeoutHandle = setTimeout(() => {
        onTimerExpiry(io, roomId)
    }, TIMER_DURATION * 1000)

    activeTimers.set(roomId, timeoutHandle)

    io.to(roomId).emit('timerStarted', {
        timerState: 'RUNNING',
        timerEndsAt
    })
}

export { startTimer, TIMER_DURATION, activeTimers, clearExistingTimer }