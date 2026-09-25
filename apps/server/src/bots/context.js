import redis from '../redis/client.js'
import { getPlayer } from './playerCache.js'

// Reads the live auction out of Redis and turns it into the plain "context"
// object every bot decides from — the exact shape the training simulator
// builds (documented in packages/shared/src/observation.js). Same input
// shape in training and in production is what makes a trained policy
// behave the same way in a real room.

const readAuctionState = async (roomId) => {
    const [room, current, teamsMap, pool, unsold] = await Promise.all([
        redis.hgetall(`room:${roomId}`),
        redis.hgetall(`room:${roomId}:current`),
        redis.hgetall(`room:${roomId}:teams`),
        redis.lrange(`room:${roomId}:pool`, 0, -1),
        // Read-only: unsold players so far. During the main round they come
        // back in the re-auction, so the planning layer counts them as supply.
        redis.lrange(`room:${roomId}:pool:unsold`, 0, -1)
    ])

    const teams = await Promise.all(
        Object.keys(teamsMap || {}).map(async (teamId) => {
            const [team, squad] = await Promise.all([
                redis.hgetall(`room:${roomId}:team:${teamId}`),
                redis.lrange(`room:${roomId}:team:${teamId}:squad`, 0, -1)
            ])
            return {
                teamId,
                ownerId: team.ownerId,
                isBot: team.isBot === 'true',
                purseLeft: Number(team.purseLeft),
                purseSpent: Number(team.purseSpent),
                playerCount: Number(team.playerCount),
                overseasCount: Number(team.overseasCount),
                squadSlNos: squad
            }
        })
    )

    return { room, current, teams, pool, unsold }
}

const rulesOf = (room) => ({
    pursePerTeam: Number(room.pursePerTeam),
    maxPlayers: Number(room.maxPlayers),
    maxOverseas: Number(room.maxOverseas)
})

// Returns (teamId) => context for the lot currently on the block.
const contextBuilder = (state, players) => {
    const { room, current, teams, pool, unsold = [] } = state
    const rules = rulesOf(room)
    const lot = getPlayer(players, current.iplPlayerId)

    const teamViews = teams.map((t) => ({
        teamId: t.teamId,
        purseLeft: t.purseLeft,
        purseSpent: t.purseSpent,
        playerCount: t.playerCount,
        overseasCount: t.overseasCount,
        squad: t.squadSlNos.map((slNo) => getPlayer(players, slNo)).filter(Boolean)
    }))

    const index = Number(room.currentPlayerIndex)
    const phase = room.auctionPhase === 'main' ? 'main' : 'reauction'
    const upcoming = pool.slice(index + 1).map((slNo) => getPlayer(players, slNo)).filter(Boolean)
    const returning = phase === 'main'
        ? unsold.filter((slNo) => slNo !== current.iplPlayerId).map((slNo) => getPlayer(players, slNo)).filter(Boolean)
        : []
    const progress = phase === 'main' && pool.length ? index / pool.length : 1

    return (teamId) => ({
        rules,
        lot,
        self: teamViews.find((t) => t.teamId === teamId),
        rivals: teamViews.filter((t) => t.teamId !== teamId),
        upcoming,
        returning,
        phase,
        progress
    })
}

export { readAuctionState, contextBuilder, rulesOf }
