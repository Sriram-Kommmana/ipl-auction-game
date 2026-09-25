import redis from '../redis/client.js'
import { FOUR_DAYS_IN_SECONDS, IPL_TEAMS } from '../constants.js'

// Claims a franchise for a player (human or bot) while the room is in the
// lobby. Shared by the selectTeam socket handler and solo-mode bot seating.
//
// Returns one of:
//   { ok: true, unchanged: true, teamId }            — already on this team
//   { ok: true, teamId, previousTeamId, totalTeams } — claimed
//   { ok: false, error }
const claimTeam = async ({ roomId, playerId, teamId, isBot = false }) => {
    if (!teamId || !IPL_TEAMS[teamId]) {
        return { ok: false, error: 'Invalid team selected' }
    }

    const [playerData, room] = await Promise.all([
        redis.hget(`room:${roomId}:players`, playerId),
        redis.hgetall(`room:${roomId}`)
    ])

    if (!playerData) {
        return { ok: false, error: 'You are no longer part of this room.' }
    }
    if (!room || Object.keys(room).length === 0) {
        return { ok: false, error: 'Room not found or has expired.' }
    }
    if (room.status !== 'lobby') {
        return { ok: false, error: 'Cannot change team after the auction has started.' }
    }

    const parts = playerData.split(':')
    const currentTeamId = parts[2]

    if (currentTeamId === teamId) {
        return { ok: true, unchanged: true, teamId }
    }

    // Atomic claim — succeeds only if no one else has claimed this teamId yet
    const claimed = await redis.hsetnx(`room:${roomId}:teams`, teamId, playerId)
    if (claimed === 0) {
        return { ok: false, error: `${IPL_TEAMS[teamId]} has already been taken by someone else.` }
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
        isBot:         isBot ? 'true' : 'false'
    })
    pipeline.expire(`room:${roomId}:team:${teamId}`, FOUR_DAYS_IN_SECONDS)

    parts[2] = teamId
    pipeline.hset(`room:${roomId}:players`, { [playerId]: parts.join(':') })
    pipeline.expire(`room:${roomId}:players`, FOUR_DAYS_IN_SECONDS)

    // Bots have no session — nothing ever logs in as them.
    if (!isBot) {
        pipeline.hset(`session:${playerId}`, { teamId })
        pipeline.expire(`session:${playerId}`, FOUR_DAYS_IN_SECONDS)
    }

    await pipeline.exec()

    // Update totalTeams count on room config (after pipeline completes)
    const totalTeams = await redis.hlen(`room:${roomId}:teams`)
    await redis.hset(`room:${roomId}`, { totalTeams })

    return { ok: true, teamId, previousTeamId: currentTeamId || null, totalTeams }
}

export { claimTeam }
