/**
 * timerManager is the single owner of all timer-related room state.
 *
 * This includes:
 * - room.status (active/paused)
 * - timerState (RUNNING/PAUSED/IDLE)
 * - timerEndsAt
 * - pausedTimeRemaining
 *
 * Responsibilities:
 * - Create, clear, pause, and resume timers
 * - Maintain timer-related Redis state
 * - Fire timer expiry
 *
 * It should NOT contain bidding, sold/unsold, or auction progression logic
 * beyond delegating to onTimerExpiry.
 */

import redis from '../redis/client.js'

const TIMER_DURATION = 30

// Module-level Map — must persist across calls for the life of the process.
// Not exported — nothing outside this file should ever touch it directly.
const activeTimers = new Map()

const clearExistingTimer = (roomId) => {
    if (activeTimers.has(roomId)) {
        clearTimeout(activeTimers.get(roomId))
        activeTimers.delete(roomId)
    }
}

// Phase 7: full expiry workflow — sold/unsold decision,
// advance to next player, re-auction phase, auction end detection
const onTimerExpiry = async (io, roomId) => {
    console.log(`[TimerManager STUB] onTimerExpiry fired for room ${roomId}`)
}

const startTimer = async (io, roomId) => {
    const now = Math.floor(Date.now() / 1000)
    const timerEndsAt = now + TIMER_DURATION

    await redis.hset(`room:${roomId}:current`, {
        timerState:          'RUNNING',
        timerEndsAt:         String(timerEndsAt),
        pausedTimeRemaining: ''
    })

    clearExistingTimer(roomId)

    const timeoutHandle = setTimeout(async () => {
        activeTimers.delete(roomId)

        try {
            await onTimerExpiry(io, roomId)
        } catch (err) {
            console.error(`[TimerManager] onTimerExpiry failed for room ${roomId}:`, err)
        }
    }, TIMER_DURATION * 1000)

    activeTimers.set(roomId, timeoutHandle)

    io.to(roomId).emit('timerStarted', {
        timerState: 'RUNNING',
        timerEndsAt
    })
}

const pauseTimer = async (io, roomId) => {
    const current = await redis.hgetall(`room:${roomId}:current`)

    if (Object.keys(current).length === 0 || current.timerState !== 'RUNNING') {
        return
    }

    const now = Math.floor(Date.now() / 1000)
    const timerEndsAt = Number(current.timerEndsAt)
    const remainingSeconds = Math.max(timerEndsAt - now, 0)

    clearExistingTimer(roomId)

    const pipeline = redis.pipeline()

    pipeline.hset(`room:${roomId}:current`, {
        timerState:          'PAUSED',
        timerEndsAt:         '',
        pausedTimeRemaining: String(remainingSeconds)
    })
    pipeline.hset(`room:${roomId}`, { status: 'paused' })

    await pipeline.exec()

    io.to(roomId).emit('timerPaused', {
        timerState:          'PAUSED',
        pausedTimeRemaining: remainingSeconds
    })
}

const resumeTimer = async (io, roomId) => {
    const current = await redis.hgetall(`room:${roomId}:current`)

    if (Object.keys(current).length === 0 || current.timerState !== 'PAUSED') {
        return
    }

    const remainingSeconds = Number(current.pausedTimeRemaining)
    const now = Math.floor(Date.now() / 1000)
    const timerEndsAt = now + remainingSeconds

    const pipeline = redis.pipeline()

    pipeline.hset(`room:${roomId}:current`, {
        timerState:          'RUNNING',
        timerEndsAt:         String(timerEndsAt),
        pausedTimeRemaining: ''
    })
    pipeline.hset(`room:${roomId}`, { status: 'active' })

    await pipeline.exec()

    clearExistingTimer(roomId)

    const timeoutHandle = setTimeout(async () => {
        activeTimers.delete(roomId)

        try {
            await onTimerExpiry(io, roomId)
        } catch (err) {
            console.error(`[TimerManager] onTimerExpiry failed for room ${roomId}:`, err)
        }
    }, remainingSeconds * 1000)

    activeTimers.set(roomId, timeoutHandle)

    io.to(roomId).emit('timerResumed', {
        timerState: 'RUNNING',
        timerEndsAt
    })
}

export {
    startTimer,
    pauseTimer,
    resumeTimer,
    TIMER_DURATION,
    clearExistingTimer
}