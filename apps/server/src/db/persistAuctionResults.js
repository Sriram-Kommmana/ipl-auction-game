import redis from '../redis/client.js'
import Player from './models/Player.js'
import AuctionResult from './models/AuctionResult.js'
import Room from './models/Room.js'
import { teamStrength } from '@ipl-auction/shared'

// teamRating is the strength of the best playing XI the squad can field
// (max 4 overseas, a keeper, 5 bowling options; empty slots count as 0) —
// see packages/shared/src/scoring.js. It used to be the plain average of the
// whole squad, which let one superstar and nothing else top the leaderboard.
const calculateTeamRating = (squadPlayers) => teamStrength(squadPlayers)

const persistAuctionResults = async (roomId) => {
    console.log(`[persistAuctionResults] Starting for room ${roomId}`)

    // Read all required Redis keys in parallel
    const [room, teamIds, historyRaw, playersRaw, botsRaw] = await Promise.all([
        redis.hgetall(`room:${roomId}`),
        redis.hkeys(`room:${roomId}:teams`),
        redis.lrange(`room:${roomId}:history`, 0, -1),
        redis.hgetall(`room:${roomId}:players`),
        redis.hgetall(`room:${roomId}:bots`)
    ])

    const botSeats = {}
    for (const [botId, raw] of Object.entries(botsRaw || {})) {
        try { botSeats[botId] = JSON.parse(raw) } catch { /* ignore corrupt seat */ }
    }

    if (!room || Object.keys(room).length === 0) {
        throw new Error(`Room ${roomId} not found in Redis`)
    }

    // Build nickname map once — playerId → nickname
    const playerNicknameMap = {}
    for (const [playerId, data] of Object.entries(playersRaw || {})) {
        const [nickname] = data.split(':')
        playerNicknameMap[playerId] = nickname
    }

    // Build soldPriceMap ONCE outside the team loop — O(history length)
    // Maps "slNo:teamId" → soldFor price
    // Keyed by both slNo and teamId to handle edge cases where the same
    // player appears in re-auction (different teamId could win)
    const soldPriceMap = {}
    for (const entry of historyRaw) {
        try {
            const parsed = JSON.parse(entry)
            if (parsed.status === 'sold') {
                soldPriceMap[`${parsed.iplPlayerId}:${parsed.soldTo}`] = parsed.soldFor
            }
        } catch { /* skip corrupted entries */ }
    }

    // Collect all squad slNos across ALL teams in one pass
    // so we can do a single MongoDB query for all players
    const teamDataMap   = {}
    const teamSquadMap  = {}
    const allSlNos      = new Set()

    for (const teamId of teamIds) {
        const [teamData, squadSlNos] = await Promise.all([
            redis.hgetall(`room:${roomId}:team:${teamId}`),
            redis.lrange(`room:${roomId}:team:${teamId}:squad`, 0, -1)
        ])

        if (!teamData || Object.keys(teamData).length === 0) {
            console.warn(`[persistAuctionResults] Team ${teamId} data missing, skipping`)
            continue
        }

        teamDataMap[teamId]  = teamData
        teamSquadMap[teamId] = squadSlNos
        for (const slNo of squadSlNos) {
            allSlNos.add(Number(slNo))
        }
    }

    // Single MongoDB query for ALL players across ALL teams
    const allPlayerDocs = await Player.find({
        slNo: { $in: [...allSlNos] }
    }).lean()

    // Index by slNo for O(1) lookup per player
    const playerDocMap = {}
    for (const doc of allPlayerDocs) {
        playerDocMap[doc.slNo] = doc
    }

    // Build teams array using pre-fetched data
    const teams = []

    for (const teamId of Object.keys(teamDataMap)) {
        const teamData   = teamDataMap[teamId]
        const squadSlNos = teamSquadMap[teamId]

        const squadPlayers = squadSlNos
            .map(slNo => {
                const doc = playerDocMap[Number(slNo)]
                if (!doc) return null
                return {
                    slNo:        doc.slNo,
                    playerName:  doc.playerName,
                    role:        doc.role,
                    nationality: doc.nationality,
                    boughtFor:   soldPriceMap[`${slNo}:${teamId}`] ?? 0,
                    rating:      doc.rating,
                    stats: {
                        bat: doc.stats?.bat ?? 0,
                        pwr: doc.stats?.pwr ?? 0,
                        bwl: doc.stats?.bwl ?? 0,
                        tec: doc.stats?.tec ?? 0,
                        clt: doc.stats?.clt ?? 0
                    }
                }
            })
            .filter(Boolean)

        teams.push({
            teamId,
            teamName:      teamData.name,
            ownerId:       teamData.ownerId,
            ownerNickname: playerNicknameMap[teamData.ownerId] || '',
            isBot:         teamData.isBot === 'true',
            botKind:       botSeats[teamData.ownerId]?.kind ?? null,
            botPersona:    botSeats[teamData.ownerId]?.persona ?? null,
            purseSpent:    Number(teamData.purseSpent),
            purseLeft:     Number(teamData.purseLeft),
            playerCount:   Number(teamData.playerCount),
            overseasCount: Number(teamData.overseasCount),
            teamRating:    calculateTeamRating(squadPlayers),
            squad:         squadPlayers
        })
    }

    // Build history array in chronological order
    const history = []
    let auctionOrder = 1

    for (const entry of historyRaw) {
        try {
            const parsed = JSON.parse(entry)
            history.push({
                auctionOrder: auctionOrder++,
                slNo:         Number(parsed.iplPlayerId),
                playerName:   parsed.playerName,
                soldTo:       parsed.soldTo  || null,
                soldFor:      parsed.soldFor || null,
                status:       parsed.status,
                auctionedAt:  new Date(parsed.soldAt * 1000)
            })
        } catch {
            console.warn(`[persistAuctionResults] Skipping corrupted history entry`)
        }
    }

    console.log(`[persistAuctionResults] Room ${roomId} — ${teams.length} teams, ${history.length} history entries`)

    const auctionResult = {
        roomId,
        version:     2,   // v2: teamRating = best-XI strength; bot owners recorded
        mode:        room.mode || 'multiplayer',
        startedAt:   new Date(Number(room.startedAt)   * 1000),
        completedAt: new Date(Number(room.completedAt) * 1000),
        teams,
        history
    }

    // Upsert — safe to call multiple times for the same room
    await AuctionResult.findOneAndUpdate(
        { roomId },
        auctionResult,
        { upsert: true, new: true }
    )

    // Mark room as completed in MongoDB
    await Room.findOneAndUpdate(
        { roomId },
        { completedAt: new Date(Number(room.completedAt) * 1000) }
    )

    console.log(`[persistAuctionResults] Successfully persisted results for room ${roomId}`)
}

export { persistAuctionResults }