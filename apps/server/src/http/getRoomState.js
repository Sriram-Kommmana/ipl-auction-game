import { ApiError } from '../utils/ApiError.js'
import { ApiResponse } from '../utils/ApiResponse.js'
import { asyncHandler } from "../utils/asyncHandler.js"
import redis from '../redis/client.js'

const getRoomState = asyncHandler(async (req, res) => {
    const { roomId } = req.params

    // Validate roomId param
    if (!roomId?.trim()) {
        throw new ApiError(400, "Room ID is required")
    }

    const cleanRoomId = roomId.trim().toUpperCase()

    // Fetch room config, players and teams in parallel
    const [room, playersRaw, teamsTaken] = await Promise.all([
        redis.hgetall(`room:${cleanRoomId}`),
        redis.hgetall(`room:${cleanRoomId}:players`),
        redis.lrange(`room:${cleanRoomId}:teams`, 0, -1)
    ])

    // Room doesn't exist
    if (!room || Object.keys(room).length === 0) {
        throw new ApiError(404, "Room not found or has expired")
    }

    // Parse players and strip sensitive fields
    const players = Object.values(playersRaw || {}).map((data) => {
        const parts = data.split(':')

        const nickname = parts[0]
        // parts[1] is pinHash — intentionally skipped
        const teamId = parts[2]
        const isManager = parts[3] === 'true'
        const status = parts[4]
        const isBot = parts[5] === 'true'

        return {
            nickname,
            teamId,
            isManager,
            status,
            isBot
        }
    })

    // Build safe response object
    const roomState = {
        roomId: room.roomId,
        status: room.status,
        auctionPhase: room.auctionPhase,
        pursePerTeam: Number(room.pursePerTeam),
        timerDuration: Number(room.timerDuration),
        maxPlayers: Number(room.maxPlayers),
        maxOverseas: Number(room.maxOverseas),
        totalTeams: teamsTaken.length,
        teamsTaken,
        players,
        playerCount: players.length
    }

    return res.status(200).json(
        new ApiResponse(
            200,
            roomState,
            "Room state fetched successfully"
        )
    )
})

export { getRoomState }