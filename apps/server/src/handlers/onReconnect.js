import redis from '../redis/client.js'
import {
    registerSocket,
    cancelGraceTimer,
    hasGraceTimer
} from '../socket/socketRegistry.js'

const parsePlayerData = (data) => {
    const parts = data.split(':')
    return {
        nickname:  parts[0],
        teamId:    parts[2],
        isManager: parts[3] === 'true',
        status:    parts[4],
        isBot:     parts[5] === 'true'
    }
}

const buildStateSnapshot = async (roomId) => {
    const [room, playersRaw, teamsMap, current, chatRaw, historyRaw] = await Promise.all([
        redis.hgetall(`room:${roomId}`),
        redis.hgetall(`room:${roomId}:players`),
        redis.hgetall(`room:${roomId}:teams`),
        redis.hgetall(`room:${roomId}:current`),
        redis.lrange(`room:${roomId}:chat`, 0, -1),
        redis.lrange(`room:${roomId}:history`, 0, -1)
    ])

    const players = Object.entries(playersRaw || {}).map(([playerId, data]) => {
        const parsed = parsePlayerData(data)
        return { playerId, ...parsed }
    })

    const teamIds = Object.keys(teamsMap || {})

    const teams = await Promise.all(
        teamIds.map(async (teamId) => {
            const [teamData, squad] = await Promise.all([
                redis.hgetall(`room:${roomId}:team:${teamId}`),
                redis.lrange(`room:${roomId}:team:${teamId}:squad`, 0, -1)
            ])
            return {
                teamId,
                name:          teamData.name,
                ownerId:       teamData.ownerId,
                purseLeft:     Number(teamData.purseLeft),
                purseSpent:    Number(teamData.purseSpent),
                playerCount:   Number(teamData.playerCount),
                overseasCount: Number(teamData.overseasCount),
                isBot:         teamData.isBot === 'true',
                squad
            }
        })
    )

    const safeParse = (entry) => {
        try { return JSON.parse(entry) } catch { return null }
    }

    const chat    = (chatRaw    || []).map(safeParse).filter(Boolean).reverse()
    const history = (historyRaw || []).map(safeParse).filter(Boolean).reverse()

    return {
        room: {
            roomId:             room.roomId,
            status:             room.status,
            auctionPhase:       room.auctionPhase,
            currentPlayerIndex: Number(room.currentPlayerIndex),
            pursePerTeam:       Number(room.pursePerTeam),
            timerDuration:      Number(room.timerDuration),
            maxPlayers:         Number(room.maxPlayers),
            maxOverseas:        Number(room.maxOverseas),
            managerPlayerId:    room.managerPlayerId
        },
        players,
        teams,
        current: {
            iplPlayerId:         current.iplPlayerId || '',
            playerName:          current.playerName || '',
            role:                current.role || '',
            nationality:         current.nationality || '',
            country:             current.country || '',
            basePrice:           current.basePrice ? Number(current.basePrice) : null,
            currentBid:          current.currentBid ? Number(current.currentBid) : null,
            currentBidderId:     current.currentBidderId || '',
            rating:              current.rating ? Number(current.rating) : null,
            stats:               current.stats ? JSON.parse(current.stats) : null,
            timerState:          current.timerState || 'IDLE',
            timerEndsAt:         current.timerEndsAt ? Number(current.timerEndsAt) : null,
            pausedTimeRemaining: current.pausedTimeRemaining ? Number(current.pausedTimeRemaining) : null
        },
        chat,
        history
    }
}

const onReconnect = async (io, socket, data) => {
    const { playerId } = data || {}

    if (!playerId) {
        return socket.emit('reconnectError', { message: 'playerId is required' })
    }

    const session = await redis.hgetall(`session:${playerId}`)
    if (!session || Object.keys(session).length === 0) {
        return socket.emit('reconnectError', {
            message: 'Session not found. Please join the room again.'
        })
    }

    const { roomId } = session

    const playerData = await redis.hget(`room:${roomId}:players`, playerId)
    if (!playerData) {
        return socket.emit('reconnectError', {
            message: 'Room not found or you are no longer part of it.'
        })
    }

    socket.join(roomId)
    registerSocket(socket.id, playerId)

    // Mark player online
    const parts = playerData.split(':')
    parts[4] = 'online'
    await redis.hset(`room:${roomId}:players`, { [playerId]: parts.join(':') })

    // If manager is reconnecting within the grace period, cancel auto-pause
    const isManager = parts[3] === 'true'
    if (isManager && hasGraceTimer(roomId)) {
        cancelGraceTimer(roomId)
        console.log(`[onReconnect] Manager reconnected for room ${roomId} — grace timer cancelled`)
        io.to(roomId).emit('managerReconnected', {
            message: 'Manager reconnected. Auction continues.'
        })
    }

    const snapshot = await buildStateSnapshot(roomId)
    socket.emit('stateSync', snapshot)

    socket.to(roomId).emit('playerOnline', {
        playerId,
        nickname: parts[0]
    })
}

export { onReconnect }