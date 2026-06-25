import redis from '../redis/client.js'
import Player from '../db/models/Player.js'
import { startTimer } from '../room/timerManager.js'

const onStartAuction = async (io, socket, data) => {
    const { playerId } = data || {}

    if (!playerId) {
        return socket.emit('startAuctionError', { message: 'playerId is required' })
    }

    const session = await redis.hgetall(`session:${playerId}`)
    if (!session || !session.roomId) {
        return socket.emit('startAuctionError', { message: 'Session not found. Please rejoin the room.' })
    }

    const { roomId } = session

    // Membership validation — consistent with onReconnect/onSelectTeam
    const isMember = await redis.hexists(`room:${roomId}:players`, playerId)
    if (!isMember) {
        return socket.emit('startAuctionError', { message: 'You are no longer part of this room.' })
    }

    const room = await redis.hgetall(`room:${roomId}`)
    if (!room || Object.keys(room).length === 0) {
        return socket.emit('startAuctionError', { message: 'Room not found or has expired.' })
    }

    if (room.managerPlayerId !== playerId) {
        return socket.emit('startAuctionError', { message: 'Only the manager can start the auction.' })
    }

    if (room.status !== 'lobby') {
        return socket.emit('startAuctionError', { message: 'Auction has already started or ended.' })
    }

    const totalTeams = await redis.hlen(`room:${roomId}:teams`)
    if (totalTeams < 2) {
        return socket.emit('startAuctionError', { message: 'At least 2 teams must be claimed before starting.' })
    }

    const firstSlNo = await redis.lindex(`room:${roomId}:pool`, 0)
    if (!firstSlNo) {
        return socket.emit('startAuctionError', { message: 'Auction pool is empty. Please contact support.' })
    }

    const playerDoc = await Player.findOne({ slNo: Number(firstSlNo) }).lean()
    if (!playerDoc) {
        return socket.emit('startAuctionError', { message: 'First player data not found in database.' })
    }

    const now = Math.floor(Date.now() / 1000)

    const pipeline = redis.pipeline()

    // Set up the auction slot — timer fields left IDLE, timerManager owns them
    pipeline.hset(`room:${roomId}:current`, {
        iplPlayerId:         String(playerDoc.slNo),
        basePrice:           String(playerDoc.basePrice),
        currentBid:          String(playerDoc.basePrice),
        currentBidderId:     '',
        timerState:          'IDLE',
        timerEndsAt:         '',
        pausedTimeRemaining: ''
    })

    pipeline.hset(`room:${roomId}`, {
        status:             'active',
        startedAt:          String(now),
        currentPlayerIndex: '0'
    })

    // pool:status stays "pending" — room:{roomId}:current is the single
    // source of truth for which player is live right now

    await pipeline.exec()

    // Only player/bid data here — no timer fields, timerManager broadcasts those separately
    io.to(roomId).emit('auctionStarted', {
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
        currentBidderId: ''
    })

    // timerManager owns all timer state and broadcasts 'timerStarted' itself
    await startTimer(io, roomId)
}

export { onStartAuction }