import redis from '../redis/client.js'

const FOUR_DAYS_IN_SECONDS = 4 * 24 * 60 * 60

const IPL_TEAMS = {
    MI:   'Mumbai Indians',
    CSK:  'Chennai Super Kings',
    RCB:  'Royal Challengers Bengaluru',
    KKR:  'Kolkata Knight Riders',
    DC:   'Delhi Capitals',
    SRH:  'Sunrisers Hyderabad',
    PBKS: 'Punjab Kings',
    RR:   'Rajasthan Royals',
    GT:   'Gujarat Titans',
    LSG:  'Lucknow Super Giants'
}

const onSelectTeam = async (io, socket, data) => {
    const { playerId, teamId } = data || {}

    if (!playerId) {
        return socket.emit('selectTeamError', { message: 'playerId is required' })
    }

    if (!teamId || !IPL_TEAMS[teamId]) {
        return socket.emit('selectTeamError', { message: 'Invalid team selected' })
    }

    const session = await redis.hgetall(`session:${playerId}`)
    if (!session || !session.roomId) {
        return socket.emit('selectTeamError', { message: 'Session not found. Please rejoin the room.' })
    }

    const { roomId } = session

    const playerData = await redis.hget(`room:${roomId}:players`, playerId)
    if (!playerData) {
        return socket.emit('selectTeamError', { message: 'You are no longer part of this room.' })
    }

    const room = await redis.hgetall(`room:${roomId}`)
    if (!room || Object.keys(room).length === 0) {
        return socket.emit('selectTeamError', { message: 'Room not found or has expired.' })
    }

    if (room.status !== 'lobby') {
        return socket.emit('selectTeamError', { message: 'Cannot change team after the auction has started.' })
    }

    const parts = playerData.split(':')
    const currentTeamId = parts[2]

    // Already on this team — nothing to do, just re-confirm
    if (currentTeamId === teamId) {
        return socket.emit('teamSelected', {
            playerId,
            teamId,
            teamName: IPL_TEAMS[teamId]
        })
    }

    // Atomic claim — succeeds only if no one else has claimed this teamId yet
    const claimed = await redis.hsetnx(`room:${roomId}:teams`, teamId, playerId)

    if (claimed === 0) {
        return socket.emit('selectTeamError', {
            message: `${IPL_TEAMS[teamId]} has already been taken by someone else.`
        })
    }

    // Claim succeeded — release old team (if switching) and set up new team
    const pipeline = redis.pipeline()

    if (currentTeamId) {
        pipeline.hdel(`room:${roomId}:teams`, currentTeamId)
        pipeline.del(`room:${roomId}:team:${currentTeamId}`)
        pipeline.del(`room:${roomId}:team:${currentTeamId}:squad`)
    }

    pipeline.expire(`room:${roomId}:teams`, FOUR_DAYS_IN_SECONDS)

    pipeline.hset(`room:${roomId}:team:${teamId}`, {
        teamId,
        name:          IPL_TEAMS[teamId],
        ownerId:       playerId,
        purseLeft:     room.pursePerTeam,
        purseSpent:    '0',
        playerCount:   '0',
        overseasCount: '0',
        isBot:         'false'
    })
    pipeline.expire(`room:${roomId}:team:${teamId}`, FOUR_DAYS_IN_SECONDS)

    parts[2] = teamId
    pipeline.hset(`room:${roomId}:players`, { [playerId]: parts.join(':') })
    pipeline.expire(`room:${roomId}:players`, FOUR_DAYS_IN_SECONDS)

    pipeline.hset(`session:${playerId}`, { teamId })
    pipeline.expire(`session:${playerId}`, FOUR_DAYS_IN_SECONDS)

    await pipeline.exec()

    // Update totalTeams count on room config (after pipeline completes)
    const totalTeams = await redis.hlen(`room:${roomId}:teams`)
    await redis.hset(`room:${roomId}`, { totalTeams })

    // Broadcast to everyone — frontend updates the team grid for all
    io.to(roomId).emit('teamSelected', {
        playerId,
        teamId,
        teamName:       IPL_TEAMS[teamId],
        previousTeamId: currentTeamId || null,
        totalTeams
    })
}

export { onSelectTeam, IPL_TEAMS }