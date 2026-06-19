import { ApiError } from '../utils/ApiError.js'
import { ApiResponse } from '../utils/ApiResponse.js'
import { asyncHandler } from "../utils/asyncHandler.js";
import redis from '../redis/client.js'

const getRoomState = asyncHandler(async (req, res) => {
    const { roomId } = req.params

    // Validate roomId param
    if (!roomId?.trim()) {
        throw new ApiError(400, "Room ID is required")
    }

    const cleanRoomId = roomId.trim().toUpperCase()

    // Check room exists
    const roomExists = await redis.exists(`room:${cleanRoomId}`)
    if (!roomExists) {
        throw new ApiError(404, "Room not found or has expired")
    }

    // Fetch room config
    const room = await redis.hgetall(`room:${cleanRoomId}`)

    if (!room || Object.keys(room).length === 0) {
        throw new ApiError(404, "Room not found or has expired")
    }

    // Fetch all players in the room
    const playersRaw = await redis.hgetall(`room:${cleanRoomId}:players`) || {}

    // Parse players, strip sensitive fields (pinHash, playerId)
    const players = Object.values(playersRaw).map((data) => {
        const parts = data.split(':')
        const nickname  = parts[0]
        // parts[1] is pinHash — intentionally skipped, never exposed
        const teamId    = parts[2]
        const isManager = parts[3] === 'true'
        const status    = parts[4]
        const isBot     = parts[5] === 'true'

        return {
            nickname,
            teamId,
            isManager,
            status,
            isBot
        }
    })

    // Fetch teams already claimed
    const teamsTaken = await redis.lrange(`room:${cleanRoomId}:teams`, 0, -1) || []

    // Build safe response object
    const roomState = {
        roomId:         room.roomId,
        status:         room.status,
        auctionPhase:   room.auctionPhase,
        pursePerTeam:   Number(room.pursePerTeam),
        timerDuration:  Number(room.timerDuration),
        maxPlayers:     Number(room.maxPlayers),
        maxOverseas:    Number(room.maxOverseas),
        totalTeams:     teamsTaken.length,
        teamsTaken,
        players,
        playerCount:    players.length
    }

    return res.status(200).json(
        new ApiResponse(200, roomState, "Room state fetched successfully")
    )
})

export { getRoomState }