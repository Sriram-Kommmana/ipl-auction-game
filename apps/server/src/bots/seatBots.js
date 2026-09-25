import { v4 as uuidv4 } from 'uuid'
import { SOLO_BOT_LINEUP, botDisplayName, shuffle } from '@ipl-auction/shared'
import redis from '../redis/client.js'
import { claimTeam } from '../services/claimTeam.js'
import { FOUR_DAYS_IN_SECONDS, IPL_TEAMS } from '../constants.js'

// Fills every unclaimed franchise in a solo room with a bot, through the same
// claimTeam path a human uses. Each bot is a real member of the room:
//   room:{id}:players  botId → "Name:::false:online:true"
//                      (no PIN hash, so nobody can ever log in as a bot;
//                       always "online"; isBot = true)
//   room:{id}:bots     botId → { kind: 'rule'|'rl', persona, teamId }
// The second hash is how the bot runtime finds its seats again, e.g. after
// a server restart.
const seatBots = async (roomId) => {
    const taken = await redis.hgetall(`room:${roomId}:teams`)
    const freeTeams = Object.keys(IPL_TEAMS).filter((teamId) => !taken?.[teamId])
    const lineup = shuffle(SOLO_BOT_LINEUP).slice(0, freeTeams.length)

    const seats = []
    for (const [i, teamId] of freeTeams.entries()) {
        const seat = { ...lineup[i], teamId }
        const botId = `bot-${uuidv4()}`
        const name = botDisplayName(seat)

        await redis.hset(`room:${roomId}:players`, { [botId]: `${name}:::false:online:true` })
        const claim = await claimTeam({ roomId, playerId: botId, teamId, isBot: true })
        if (!claim.ok) {
            await redis.hdel(`room:${roomId}:players`, botId)
            throw new Error(`Could not seat bot on ${teamId}: ${claim.error}`)
        }

        await redis.hset(`room:${roomId}:bots`, { [botId]: JSON.stringify(seat) })
        seats.push({ botId, ...seat })
    }
    await redis.expire(`room:${roomId}:bots`, FOUR_DAYS_IN_SECONDS)

    return seats
}

export { seatBots }
