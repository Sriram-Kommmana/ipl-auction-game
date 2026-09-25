import redis from '../redis/client.js'
import { armTimer, getTimerSeconds, newTimerToken } from '../room/timerManager.js'
import { emitToRoom } from './roomEvents.js'

// Places one bid (always exactly one increment — the amount is computed by
// the server, never chosen by the bidder). Shared by human sockets and
// server-side bots, so both go through identical validation.
//
// expectedAmount (optional): the price the bidder decided to pay. If someone
// else bid first, the price has moved and the bid is rejected as `stale`
// instead of silently charging the next increment — bots rely on this so
// they can never pay above their cap.
//
// Returns { success, newBid } or { success: false, error, stale? }.
const placeBid = async (io, { roomId, teamId, expectedAmount = null }) => {
    const seconds     = await getTimerSeconds(roomId)
    const timerEndsAt = Math.floor(Date.now() / 1000) + seconds
    const token       = newTimerToken()

    let result
    try {
        const raw = await redis.placeBidAtomic(
            `room:${roomId}:current`,
            `room:${roomId}:team:${teamId}`,
            `room:${roomId}`,
            teamId,
            expectedAmount === null ? '' : String(expectedAmount),
            String(timerEndsAt),
            token
        )
        result = JSON.parse(raw)
    } catch (err) {
        console.error('[placeBid] Lua script execution failed:', err)
        return { success: false, error: 'Something went wrong placing your bid. Please try again.' }
    }

    if (!result.success) return result

    // The bid and its fresh countdown were written in one atomic step;
    // now schedule the matching local timeout and tell everyone.
    armTimer(io, roomId, token, seconds * 1000)

    emitToRoom(io, roomId, 'timerStarted', {
        timerState: 'RUNNING',
        timerEndsAt
    })
    emitToRoom(io, roomId, 'bidPlaced', {
        teamId,
        newBid: result.newBid
    })

    return result
}

export { placeBid }
