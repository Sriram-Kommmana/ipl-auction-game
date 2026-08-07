/**
 * timerManager is the single owner of all timer-related room state.
 *
 * This includes:
 * - room.status (active/paused)
 * - timerState (RUNNING/PAUSED/IDLE/PROCESSING_EXPIRY/ENDED)
 * - timerEndsAt
 * - pausedTimeRemaining
 *
 * Responsibilities:
 * - Create, clear, pause, and resume timers
 * - Maintain timer-related Redis state
 * - Fire timer expiry — decide sold/unsold, then delegate
 *   advancement to auctionProgression.advanceAuction()
 *
 * It should NOT contain bid validation, session handling,
 * or socket membership logic — those belong in handlers.
 */

import { v4 as uuidv4 } from 'uuid'
import redis from '../redis/client.js'
import { advanceAuction } from './auctionProgression.js'
import {RESULT_DISPLAY_DURATION} from '../constants.js'

const TIMER_DURATION = 30
const FOUR_DAYS_IN_SECONDS = 4 * 24 * 60 * 60

const delay = (ms) =>{
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Module-level Map — must persist across calls for the life of the process.
// Not exported — nothing outside this file should ever touch it directly.
const activeTimers = new Map()

const clearExistingTimer = (roomId) => {
    if (activeTimers.has(roomId)) {
        clearTimeout(activeTimers.get(roomId))
        activeTimers.delete(roomId)
    }
}

const onTimerExpiry = async (io, roomId) => {
    const [room, current, poolLength] = await Promise.all([
        redis.hgetall(`room:${roomId}`),
        redis.hgetall(`room:${roomId}:current`),
        redis.llen(`room:${roomId}:pool`)
    ])

    if (!current || Object.keys(current).length === 0) {
        console.log(`[TimerManager] onTimerExpiry: no current state for room ${roomId}`)
        return
    }

    // Atomically acquire the lock — returns 0 if onSkip already grabbed it
    // or if timerState is already past RUNNING/PAUSED
    const acquired = await redis.acquireExpiryLock(`room:${roomId}:current`)
    if (!acquired) {
        console.log(`[TimerManager] onTimerExpiry: lock not acquired for room ${roomId}, skipping`)
        return
    }

    const {
        iplPlayerId,
        playerName,
        nationality,
        role,
        currentBid,
        currentBidderId
    } = current

    const currentPlayerIndex = Number(room.currentPlayerIndex)
    const auctionPhase       = room.auctionPhase
    const isSold             = currentBidderId !== ''

    try {
        if (isSold) {
            const teamKey  = `room:${roomId}:team:${currentBidderId}`
            const teamData = await redis.hgetall(teamKey)

            if (!teamData || Object.keys(teamData).length === 0) {
                throw new Error(`Team data not found for ${currentBidderId} in room ${roomId}`)
            }

            const newPurseLeft     = Number(teamData.purseLeft) - Number(currentBid)
            const newPurseSpent    = Number(teamData.purseSpent) + Number(currentBid)
            const newPlayerCount   = Number(teamData.playerCount) + 1
            const newOverseasCount = nationality === 'Overseas'
                ? Number(teamData.overseasCount) + 1
                : Number(teamData.overseasCount)

            const soldAt = Math.floor(Date.now() / 1000)

            const historyEntry = JSON.stringify({
                iplPlayerId,
                playerName,
                role,
                soldTo:  currentBidderId,
                soldFor: Number(currentBid),
                status:  'sold',
                soldAt
            })

            const chatEntry = JSON.stringify({
                messageId: uuidv4(),
                playerId:  'system',
                nickname:  'Auction',
                type:      'broadcast',
                text:      `${playerName} sold to ${teamData.name} for ₹${currentBid}L`,
                sentAt:    soldAt
            })

            const pipeline = redis.pipeline()
            pipeline.hset(teamKey, {
                purseLeft:     String(newPurseLeft),
                purseSpent:    String(newPurseSpent),
                playerCount:   String(newPlayerCount),
                overseasCount: String(newOverseasCount)
            })
            pipeline.rpush(`room:${roomId}:team:${currentBidderId}:squad`, iplPlayerId)
            pipeline.expire(`room:${roomId}:team:${currentBidderId}:squad`, FOUR_DAYS_IN_SECONDS)
            pipeline.hset(`room:${roomId}:pool:status`, { [iplPlayerId]: 'sold' })
            pipeline.rpush(`room:${roomId}:history`, historyEntry)
            pipeline.expire(`room:${roomId}:history`, FOUR_DAYS_IN_SECONDS)
            pipeline.lpush(`room:${roomId}:chat`, chatEntry)
            pipeline.ltrim(`room:${roomId}:chat`, 0, 199)
            pipeline.expire(`room:${roomId}:chat`, FOUR_DAYS_IN_SECONDS)
            await pipeline.exec()

            io.to(roomId).emit('playerSold', {
                iplPlayerId,
                playerName,
                role,
                soldTo:     currentBidderId,
                teamName:   teamData.name,
                soldFor:    Number(currentBid),
                isOverseas: nationality === 'Overseas',
                soldAt
            })

        } else {
            const soldAt = Math.floor(Date.now() / 1000)

            const historyEntry = JSON.stringify({
                iplPlayerId,
                playerName,
                soldTo:  null,
                soldFor: null,
                status:  'unsold',
                soldAt
            })

            const chatEntry = JSON.stringify({
                messageId: uuidv4(),
                playerId:  'system',
                nickname:  'Auction',
                type:      'broadcast',
                text:      `${playerName} went unsold`,
                sentAt:    soldAt
            })

            const pipeline = redis.pipeline()
            pipeline.hset(`room:${roomId}:pool:status`, { [iplPlayerId]: 'unsold' })
            pipeline.rpush(`room:${roomId}:pool:unsold`, iplPlayerId)
            pipeline.expire(`room:${roomId}:pool:unsold`, FOUR_DAYS_IN_SECONDS)
            pipeline.rpush(`room:${roomId}:history`, historyEntry)
            pipeline.expire(`room:${roomId}:history`, FOUR_DAYS_IN_SECONDS)
            pipeline.lpush(`room:${roomId}:chat`, chatEntry)
            pipeline.ltrim(`room:${roomId}:chat`, 0, 199)
            pipeline.expire(`room:${roomId}:chat`, FOUR_DAYS_IN_SECONDS)
            await pipeline.exec()

            io.to(roomId).emit('playerUnsold', {
                iplPlayerId,
                playerName,
                soldAt
            })
        }

        await delay(RESULT_DISPLAY_DURATION)

        await advanceAuction(io, roomId, currentPlayerIndex, auctionPhase, poolLength, startTimer)

    } catch (err) {
        console.error(`[TimerManager] onTimerExpiry failed for room ${roomId}:`, err)
        io.to(roomId).emit('auctionError', {
            message: 'A critical error occurred in the auction. Please contact support.'
        })
        // timerState intentionally left as PROCESSING_EXPIRY —
        // honest signal for recovery tooling, not a fake ENDED state
    }
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

// Returns true if it actually paused something, false if it silently no-op'd
// (e.g. timerState wasn't RUNNING — see onDisconnect.js's grace-timer retry
// logic, which depends on this to avoid emitting a false "paused" message).
const pauseTimer = async (io, roomId) => {
    const current = await redis.hgetall(`room:${roomId}:current`)

    if (Object.keys(current).length === 0 || current.timerState !== 'RUNNING') {
        return false
    }

    const now              = Math.floor(Date.now() / 1000)
    const timerEndsAt      = Number(current.timerEndsAt)
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

    return true
}

const resumeTimer = async (io, roomId) => {
    const current = await redis.hgetall(`room:${roomId}:current`)

    if (Object.keys(current).length === 0 || current.timerState !== 'PAUSED') {
        return
    }

    const remainingSeconds = Number(current.pausedTimeRemaining)
    const now              = Math.floor(Date.now() / 1000)
    const timerEndsAt      = now + remainingSeconds

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