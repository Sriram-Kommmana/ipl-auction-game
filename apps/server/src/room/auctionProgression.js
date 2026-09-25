import redis from '../redis/client.js'
import { loadPlayerIntoCurrent } from './loadPlayerIntoCurrent.js'
import { persistAuctionResults } from '../db/persistAuctionResults.js'
import { emitToRoom } from '../services/roomEvents.js'
import { FOUR_DAYS_IN_SECONDS } from '../constants.js'

const shuffleArray = (array) => {
    const arr = [...array]
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]]
    }
    return arr
}

const buildPlayerPayload = (playerDoc, index, extra = {}) => ({
    currentPlayerIndex: index,
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
    ...extra
})

const completeAuction = async (io, roomId, message) => {
    await redis.hset(`room:${roomId}`, {
        status:      'completed',
        completedAt: String(Math.floor(Date.now() / 1000))
    })
    await redis.hset(`room:${roomId}:current`, { timerState: 'ENDED' })

    emitToRoom(io, roomId, 'auctionCompleted', { roomId, message })

    // Fire and forget — never block auctionCompleted waiting for MongoDB
    persistAuctionResults(roomId).catch(err =>
        console.error(`[persistAuctionResults] Failed for room ${roomId}:`, err)
    )
}

const advanceAuction = async (io, roomId, currentPlayerIndex, auctionPhase, poolLength, startTimerFn) => {
    const newIndex = currentPlayerIndex + 1

    if (newIndex < poolLength) {
        // More players left in current phase
        const nextSlNo = await redis.lindex(`room:${roomId}:pool`, newIndex)
        await redis.hset(`room:${roomId}`, { currentPlayerIndex: String(newIndex) })
        const playerDoc = await loadPlayerIntoCurrent(roomId, nextSlNo)
        emitToRoom(io, roomId, 'nextPlayer', buildPlayerPayload(playerDoc, newIndex))
        await startTimerFn(io, roomId)

    } else if (auctionPhase === 'main') {
        // Main pool exhausted — check for unsold/skipped players to re-auction.
        const unsoldList = await redis.lrange(`room:${roomId}:pool:unsold`, 0, -1)

        if (unsoldList.length > 0) {
            const shuffled = shuffleArray(unsoldList)

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
            emitToRoom(io, roomId, 'nextPlayer', buildPlayerPayload(playerDoc, 0, { isReauction: true }))
            await startTimerFn(io, roomId)

        } else {
            // No unsold players — auction complete
            await completeAuction(io, roomId, 'Auction has ended. All players have been auctioned.')
        }

    } else {
        // Re-auction phase exhausted — fully complete
        await completeAuction(io, roomId, 'Auction has ended. Re-auction complete.')
    }
}

export { advanceAuction, shuffleArray, buildPlayerPayload }