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
 * - Fire timer expiry — decide sold/unsold, advance to next player,
 *   handle re-auction phase transition, detect auction completion
 *
 * It should NOT contain bid validation, session handling,
 * or socket membership logic — those belong in handlers.
 */

import { v4 as uuidv4 } from 'uuid'
import redis from '../redis/client.js'
import { loadPlayerIntoCurrent } from './loadPlayerIntoCurrent.js'

const TIMER_DURATION       = 30
const FOUR_DAYS_IN_SECONDS = 4 * 24 * 60 * 60

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
    // Read everything needed in parallel
    const [room, current, poolLength] = await Promise.all([
        redis.hgetall(`room:${roomId}`),
        redis.hgetall(`room:${roomId}:current`),
        redis.llen(`room:${roomId}:pool`)
    ])

    // Guard — if timerState is not RUNNING, something already handled this
    if (!current || current.timerState !== 'RUNNING') {
        console.log(`[TimerManager] onTimerExpiry: room ${roomId} timerState is ${current?.timerState}, skipping`)
        return
    }

    // Lock immediately — flip to PROCESSING_EXPIRY FIRST.
    // This single write prevents any further bids from being accepted
    // because the Lua script rejects bids when timerState !== 'RUNNING'.
    // timerState stays as PROCESSING_EXPIRY if anything below fails —
    // that's an honest recoverable signal, not a fake ENDED state.
    await redis.hset(`room:${roomId}:current`, {
        timerState:  'PROCESSING_EXPIRY',
        timerEndsAt: ''
    })

    const {
        iplPlayerId,
        playerName,
        nationality,
        currentBid,
        currentBidderId
    } = current

    const currentPlayerIndex = Number(room.currentPlayerIndex)
    const auctionPhase       = room.auctionPhase
    const isSold             = currentBidderId !== ''

    // Everything after the lock is wrapped in one try/catch —
    // sold pipeline, unsold pipeline, AND advancement all need protection.
    // If the sold pipeline fails halfway, the room should not be left in
    // PROCESSING_EXPIRY with no recovery path and no error surfaced.
    try {
        if (isSold) {
            const teamKey  = `room:${roomId}:team:${currentBidderId}`
            const teamData = await redis.hgetall(teamKey)

            // Guard against missing team data — prevents NaN from
            // propagating into purse/squad calculations silently
            if (!teamData || Object.keys(teamData).length === 0) {
                throw new Error(`Team data not found for ${currentBidderId} in room ${roomId}`)
            }

            const newPurseLeft     = Number(teamData.purseLeft) - Number(currentBid)
            const newPurseSpent    = Number(teamData.purseSpent) + Number(currentBid)
            const newPlayerCount   = Number(teamData.playerCount) + 1
            const newOverseasCount = nationality === 'Overseas'
                ? Number(teamData.overseasCount) + 1
                : Number(teamData.overseasCount)

            const historyEntry = JSON.stringify({
                iplPlayerId,
                playerName,
                soldTo:  currentBidderId,
                soldFor: Number(currentBid),
                status:  'sold',
                soldAt:  Math.floor(Date.now() / 1000)
            })

            const chatEntry = JSON.stringify({
                messageId: uuidv4(),
                playerId:  'system',
                nickname:  'Auction',
                type:      'broadcast',
                text:      `${playerName} sold to ${teamData.name} for ₹${currentBid}L`,
                sentAt:    Math.floor(Date.now() / 1000)
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
                soldTo:   currentBidderId,
                teamName: teamData.name,
                soldFor:  Number(currentBid)
            })

        } else {
            const historyEntry = JSON.stringify({
                iplPlayerId,
                playerName,
                soldTo:  null,
                soldFor: null,
                status:  'unsold',
                soldAt:  Math.floor(Date.now() / 1000)
            })

            const chatEntry = JSON.stringify({
                messageId: uuidv4(),
                playerId:  'system',
                nickname:  'Auction',
                type:      'broadcast',
                text:      `${playerName} went unsold`,
                sentAt:    Math.floor(Date.now() / 1000)
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
                playerName
            })
        }

        // Advance to next player
        const newIndex = currentPlayerIndex + 1

        if (newIndex < poolLength) {
            // More players left in current phase
            const nextSlNo = await redis.lindex(`room:${roomId}:pool`, newIndex)

            await redis.hset(`room:${roomId}`, {
                currentPlayerIndex: String(newIndex)
            })

            const playerDoc = await loadPlayerIntoCurrent(roomId, nextSlNo)

            io.to(roomId).emit('nextPlayer', {
                currentPlayerIndex: newIndex,
                player: {
                    slNo:        playerDoc.slNo,
                    playerName:  playerDoc.playerName,
                    country:     playerDoc.country,
                    nationality: playerDoc.nationality,
                    role:        playerDoc.role,
                    basePrice:   playerDoc.basePrice,
                    rating:      playerDoc.rating,
                    stats:       playerDoc.stats
                },
                currentBid:      playerDoc.basePrice,
                currentBidderId: ''
            })

            await startTimer(io, roomId)

        } else if (auctionPhase === 'main') {
            // Main pool exhausted — check for unsold players to re-auction
            const unsoldList = await redis.lrange(`room:${roomId}:pool:unsold`, 0, -1)

            if (unsoldList.length > 0) {
                // Shuffle the unsold list (Fisher-Yates)
                const shuffled = [...unsoldList]
                for (let i = shuffled.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
                }

                // Reset pool:status back to 'pending' for re-auctioned players
                const statusUpdates = {}
                for (const slNo of shuffled) {
                    statusUpdates[slNo] = 'pending'
                }

                const pipeline = redis.pipeline()
                pipeline.del(`room:${roomId}:pool`)
                pipeline.rpush(`room:${roomId}:pool`, ...shuffled)
                pipeline.expire(`room:${roomId}:pool`, FOUR_DAYS_IN_SECONDS)
                pipeline.hset(`room:${roomId}:pool:status`, statusUpdates)
                pipeline.del(`room:${roomId}:pool:unsold`)
                pipeline.hset(`room:${roomId}`, {
                    auctionPhase:       'reauction',
                    currentPlayerIndex: '0'
                })
                await pipeline.exec()

                const playerDoc = await loadPlayerIntoCurrent(roomId, shuffled[0])

                io.to(roomId).emit('nextPlayer', {
                    currentPlayerIndex: 0,
                    player: {
                        slNo:        playerDoc.slNo,
                        playerName:  playerDoc.playerName,
                        country:     playerDoc.country,
                        nationality: playerDoc.nationality,
                        role:        playerDoc.role,
                        basePrice:   playerDoc.basePrice,
                        rating:      playerDoc.rating,
                        stats:       playerDoc.stats
                    },
                    currentBid:      playerDoc.basePrice,
                    currentBidderId: '',
                    isReauction:     true
                })

                await startTimer(io, roomId)

            } else {
                // Main pool exhausted, no unsold players — auction complete
                await redis.hset(`room:${roomId}`, {
                    status:      'completed',
                    completedAt: String(Math.floor(Date.now() / 1000))
                })
                await redis.hset(`room:${roomId}:current`, { timerState: 'ENDED' })

                io.to(roomId).emit('auctionCompleted', {
                    roomId,
                    message: 'Auction has ended. All players have been auctioned.'
                })
            }

        } else {
            // Re-auction phase exhausted — auction fully complete
            await redis.hset(`room:${roomId}`, {
                status:      'completed',
                completedAt: String(Math.floor(Date.now() / 1000))
            })
            await redis.hset(`room:${roomId}:current`, { timerState: 'ENDED' })

            io.to(roomId).emit('auctionCompleted', {
                roomId,
                message: 'Auction has ended. Re-auction complete.'
            })
        }

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

const pauseTimer = async (io, roomId) => {
    const current = await redis.hgetall(`room:${roomId}:current`)

    if (Object.keys(current).length === 0 || current.timerState !== 'RUNNING') {
        return
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