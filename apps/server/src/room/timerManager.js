/**
 * timerManager is the single owner of all timer-related room state.
 *
 * This includes:
 * - room.status (active/paused)
 * - timerState (RUNNING/PAUSED/IDLE/PROCESSING_EXPIRY/ENDED)
 * - timerEndsAt
 * - pausedTimeRemaining
 * - timerToken
 *
 * Responsibilities:
 * - Create, clear, pause, and resume timers
 * - Maintain timer-related Redis state
 * - Fire timer expiry — decide sold/unsold, then delegate
 *   advancement to auctionProgression.advanceAuction()
 *
 * It should NOT contain bid validation, session handling,
 * or socket membership logic — those belong in handlers.
 *
 * Timer tokens: every armed timer gets a fresh token, stored in
 * room:{id}:current. When a timeout fires it may only close the lot if its
 * token is still current (checked atomically by acquireExpiryLock). A bid
 * issues a new token in the same atomic step that accepts it, so a timeout
 * that fires in the same instant as a late bid can no longer sell the lot
 * at the old price or to the old leader.
 */

import { v4 as uuidv4 } from 'uuid'
import redis from '../redis/client.js'
import { advanceAuction } from './auctionProgression.js'
import { emitToRoom } from '../services/roomEvents.js'
import {
    RESULT_DISPLAY_DURATION,
    FOUR_DAYS_IN_SECONDS,
    MULTIPLAYER_TIMER_SECONDS
} from '../constants.js'

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

const newTimerToken = () => uuidv4()

// Seconds a lot stays open after the latest bid — per room (solo rooms are
// shorter). Falls back to the multiplayer default for rooms created before
// timerDuration was honoured.
const getTimerSeconds = async (roomId) => {
    const raw = await redis.hget(`room:${roomId}`, 'timerDuration')
    const seconds = Number(raw)
    return Number.isFinite(seconds) && seconds > 0 ? seconds : MULTIPLAYER_TIMER_SECONDS
}

// Schedule the expiry for a timer whose state + token are already in Redis.
const armTimer = (io, roomId, token, ms) => {
    clearExistingTimer(roomId)
    const timeoutHandle = setTimeout(async () => {
        activeTimers.delete(roomId)
        try {
            await onTimerExpiry(io, roomId, { token })
        } catch (err) {
            console.error(`[TimerManager] onTimerExpiry failed for room ${roomId}:`, err)
        }
    }, ms)
    activeTimers.set(roomId, timeoutHandle)
}

// Returns true if this call closed the lot, false if something else had
// already closed it (or a newer timer superseded this one).
//   token          — only close if this timer is still the current one
//   requireRunning — refuse while paused (used by solo auto-close)
const onTimerExpiry = async (io, roomId, { token = '', requireRunning = false } = {}) => {
    // Lock FIRST, read SECOND. Reading the lot before locking let a bid land
    // in between: the bid was accepted and broadcast, but the lot was then
    // sold using the stale read — to the previous leader, at the previous price.
    const acquired = await redis.acquireExpiryLock(
        `room:${roomId}:current`,
        token || '',
        requireRunning ? 'running' : ''
    )
    if (!acquired) {
        return false
    }

    // Whoever closes the lot also disarms any countdown still scheduled for it.
    clearExistingTimer(roomId)

    const [room, current, poolLength] = await Promise.all([
        redis.hgetall(`room:${roomId}`),
        redis.hgetall(`room:${roomId}:current`),
        redis.llen(`room:${roomId}:pool`)
    ])

    if (!current || Object.keys(current).length === 0) {
        console.log(`[TimerManager] onTimerExpiry: no current state for room ${roomId}`)
        return false
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

            const chatObj = {
                messageId: uuidv4(),
                playerId:  'system',
                nickname:  'Auction',
                type:      'broadcast',
                text:      `${playerName} sold to ${teamData.name} for ₹${currentBid}L`,
                sentAt:    soldAt
            }
            const chatEntry = JSON.stringify(chatObj)

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

            emitToRoom(io, roomId, 'playerSold', {
                iplPlayerId,
                playerName,
                role,
                soldTo:     currentBidderId,
                teamName:   teamData.name,
                soldFor:    Number(currentBid),
                isOverseas: nationality === 'Overseas',
                soldAt
            })

            emitToRoom(io, roomId, 'newChatMessage', chatObj)

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

            const chatObj = {
                messageId: uuidv4(),
                playerId:  'system',
                nickname:  'Auction',
                type:      'broadcast',
                text:      `${playerName} went unsold`,
                sentAt:    soldAt
            }
            const chatEntry = JSON.stringify(chatObj)

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

            emitToRoom(io, roomId, 'playerUnsold', {
                iplPlayerId,
                playerName,
                soldAt
            })

            emitToRoom(io, roomId, 'newChatMessage', chatObj)
        }

        await delay(RESULT_DISPLAY_DURATION)

        await advanceAuction(io, roomId, currentPlayerIndex, auctionPhase, poolLength, startTimer)

    } catch (err) {
        console.error(`[TimerManager] onTimerExpiry failed for room ${roomId}:`, err)
        emitToRoom(io, roomId, 'auctionError', {
            message: 'A critical error occurred in the auction. Please contact support.'
        })
        // timerState intentionally left as PROCESSING_EXPIRY —
        // honest signal for recovery tooling, not a fake ENDED state
    }
    return true
}

// Opens a fresh countdown for the lot on the block (new lot, or auction start).
const startTimer = async (io, roomId) => {
    const seconds     = await getTimerSeconds(roomId)
    const now         = Math.floor(Date.now() / 1000)
    const timerEndsAt = now + seconds
    const token       = newTimerToken()

    await redis.hset(`room:${roomId}:current`, {
        timerState:          'RUNNING',
        timerEndsAt:         String(timerEndsAt),
        pausedTimeRemaining: '',
        timerToken:          token
    })

    armTimer(io, roomId, token, seconds * 1000)

    emitToRoom(io, roomId, 'timerStarted', {
        timerState: 'RUNNING',
        timerEndsAt
    })
}

// Ends the current lot right now instead of waiting for the countdown —
// used by solo mode once nobody will bid again. Never closes a paused lot.
const closeLotNow = (io, roomId) => onTimerExpiry(io, roomId, { requireRunning: true })

// Returns true if it actually paused something, false if it silently no-op'd
// (e.g. timerState wasn't RUNNING — see onDisconnect.js's grace-timer retry
// logic, which depends on this to avoid emitting a false "paused" message).
const pauseTimer = async (io, roomId) => {
    const remainingSeconds = Number(await redis.pauseTimerAtomic(
        `room:${roomId}:current`,
        `room:${roomId}`,
        String(Math.floor(Date.now() / 1000))
    ))

    if (remainingSeconds < 0) {
        return false
    }

    // Any timeout still armed carries a token the script just cleared, so it
    // can't close the lot; clearing it here just frees the handle.
    clearExistingTimer(roomId)

    emitToRoom(io, roomId, 'timerPaused', {
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
    const token            = newTimerToken()

    const pipeline = redis.pipeline()
    pipeline.hset(`room:${roomId}:current`, {
        timerState:          'RUNNING',
        timerEndsAt:         String(timerEndsAt),
        pausedTimeRemaining: '',
        timerToken:          token
    })
    pipeline.hset(`room:${roomId}`, { status: 'active' })
    await pipeline.exec()

    armTimer(io, roomId, token, remainingSeconds * 1000)

    emitToRoom(io, roomId, 'timerResumed', {
        timerState: 'RUNNING',
        timerEndsAt
    })
}

export {
    startTimer,
    pauseTimer,
    resumeTimer,
    closeLotNow,
    clearExistingTimer,
    armTimer,
    getTimerSeconds,
    newTimerToken
}
