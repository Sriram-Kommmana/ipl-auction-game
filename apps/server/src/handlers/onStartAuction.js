import redis from '../redis/client.js'
import { startTimer } from '../room/timerManager.js'
import { loadPlayerIntoCurrent } from '../room/loadPlayerIntoCurrent.js'

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

    const now = Math.floor(Date.now() / 1000)

    // Both Redis writes are independent — run them concurrently
    // loadPlayerIntoCurrent writes to room:{roomId}:current
    // the second write updates room:{roomId} status/startedAt/index
    // no ordering dependency between them, Promise.all saves one
    // network round trip versus awaiting them sequentially
    const [playerDoc] = await Promise.all([
        loadPlayerIntoCurrent(roomId, firstSlNo),
        redis.hset(`room:${roomId}`, {
            status:             'active',
            startedAt:          String(now),
            currentPlayerIndex: '0'
        })
    ])

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

    await startTimer(io, roomId)
}

export { onStartAuction }