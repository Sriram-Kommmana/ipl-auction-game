import Player from '../db/models/Player.js'
import redis from '../redis/client.js'

const loadPlayerIntoCurrent = async (roomId, slNo) => {
    const playerDoc = await Player.findOne({
        slNo: Number(slNo)
    }).lean()

    if (!playerDoc) {
        throw new Error(`Player ${slNo} not found`)
    }

    await redis.hset(`room:${roomId}:current`, {
        iplPlayerId:         String(playerDoc.slNo),
        playerName:          playerDoc.playerName,
        role:                playerDoc.role,
        nationality:         playerDoc.nationality,
        country:             playerDoc.country,
        rating:              String(playerDoc.rating),
        stats:               JSON.stringify(playerDoc.stats),
        basePrice:           String(playerDoc.basePrice),
        currentBid:          String(playerDoc.basePrice),
        currentBidderId:     '',
        timerState:          'IDLE',
        timerEndsAt:         '',
        pausedTimeRemaining: ''
    })

    return playerDoc
}

export { loadPlayerIntoCurrent }