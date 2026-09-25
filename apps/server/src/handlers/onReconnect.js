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
    const [room, playersRaw, teamsMap, current, chatRaw, historyRaw, botsRaw] = await Promise.all([
        redis.hgetall(`room:${roomId}`),
        redis.hgetall(`room:${roomId}:players`),
        redis.hgetall(`room:${roomId}:teams`),
        redis.hgetall(`room:${roomId}:current`),
        redis.lrange(`room:${roomId}:chat`, 0, -1),
        redis.lrange(`room:${roomId}:history`, 0, -1),
        redis.hgetall(`room:${roomId}:bots`)
    ])

    const botSeats = {}
    for (const [botId, raw] of Object.entries(botsRaw || {})) {
        try { botSeats[botId] = JSON.parse(raw) } catch { /* ignore corrupt seat */ }
    }

    const players = Object.entries(playersRaw || {}).map(([playerId, data]) => {
        const parsed = parsePlayerData(data)
        const seat = botSeats[playerId]
        return seat
            ? { playerId, ...parsed, botKind: seat.kind, botPersona: seat.persona }
            : { playerId, ...parsed }
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
            mode:               room.mode || 'multiplayer',
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

    const isManager = parts[3] === 'true'

    if (isManager) {
        if (hasGraceTimer(roomId)) {
            // Manager reconnected WHILE the grace period was still counting
            // down — cancel it, auction never actually paused.
            cancelGraceTimer(roomId)
            console.log(`[onReconnect] Manager reconnected for room ${roomId} — grace timer cancelled`)
            io.to(roomId).emit('managerReconnected', {
                message: 'Manager reconnected. Auction continues.'
            })
        } else {
            // Grace period may have ALREADY expired and auto-paused the
            // auction before the manager came back. hasGraceTimer() is
            // false in that case too (the timer deletes itself once it
            // fires), so we can't distinguish "no timer was ever running"
            // from "timer already fired" without checking room status
            // directly. Only emit if the room is actually paused — this
            // is what clears the "manager did not reconnect in time"
            // notice on clients. Deliberately does NOT auto-resume the
            // timer; the manager must press Resume explicitly.
            const roomStatus = await redis.hget(`room:${roomId}`, 'status')
            if (roomStatus === 'paused') {
                console.log(`[onReconnect] Manager reconnected for room ${roomId} after auto-pause`)
                io.to(roomId).emit('managerReconnected', {
                    message: 'Manager reconnected. Auction is paused — resume when ready.'
                })
            }
        }
    }

    const snapshot = await buildStateSnapshot(roomId)
    socket.emit('stateSync', snapshot)

    socket.to(roomId).emit('playerOnline', {
        playerId,
        nickname: parts[0]
    })
}

export { onReconnect, buildStateSnapshot }