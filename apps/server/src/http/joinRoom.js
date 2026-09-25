import { ApiError } from '../utils/ApiError.js'
import { ApiResponse } from '../utils/ApiResponse.js'
import { asyncHandler } from "../utils/asyncHandler.js";
import crypto from 'crypto'
import { v4 as uuidv4 } from 'uuid'
import redis from '../redis/client.js'

const FOUR_DAYS_IN_SECONDS = 4 * 24 * 60 * 60
const MAX_PLAYERS_PER_ROOM = 10

const hashPin = (pin) => crypto.createHash('sha256').update(pin).digest('hex')

const joinRoom = asyncHandler(async (req, res) => {
    const {
        roomId,
        roomPin,
        nickname,
        playerPin
    } = req.body

    if (!roomId?.trim())
        throw new ApiError(400, "Room ID is required")
    if (!roomPin)
        throw new ApiError(400, "Room PIN is required")
    if (!/^\d{4}$/.test(String(roomPin)))
        throw new ApiError(400, "Room PIN must be exactly 4 digits")
    if (!nickname?.trim())
        throw new ApiError(400, "Nickname is required")
    if (nickname.trim().length < 2)
        throw new ApiError(400, "Nickname must be at least 2 characters")
    if (nickname.trim().length > 20)
        throw new ApiError(400, "Nickname must be at most 20 characters")
    if (nickname.trim().includes(':'))
        throw new ApiError(400, "Nickname cannot contain ':'")
    if (!playerPin)
        throw new ApiError(400, "Player PIN is required")
    if (!/^\d{4}$/.test(String(playerPin)))
        throw new ApiError(400, "Player PIN must be exactly 4 digits")

    const roomExists = await redis.exists(`room:${roomId}`)
    if (!roomExists)
        throw new ApiError(404, "Room not found or has expired")

    const room = await redis.hgetall(`room:${roomId}`)

    if (room.mode === 'solo')
        throw new ApiError(403, "This is a single-player room — it can't be joined")

    const incomingRoomPinHash = hashPin(String(roomPin))
    if (incomingRoomPinHash !== room.roomPinHash)
        throw new ApiError(401, "Incorrect room PIN")

    const players = await redis.hgetall(`room:${roomId}:players`) || {}

    const playerPinHash = hashPin(String(playerPin))
    const cleanNickname  = nickname.trim()

    let existingPlayerId   = null
    let existingPlayerData = null

    for (const [pid, data] of Object.entries(players)) {
        const parts            = data.split(':')
        const existingNickname = parts[0]
        const existingPinHash  = parts[1]

        if (existingNickname.toLowerCase() === cleanNickname.toLowerCase()) {
            if (existingPinHash !== playerPinHash)
                throw new ApiError(401, "Incorrect player PIN")

            existingPlayerId   = pid
            existingPlayerData = parts
            break
        }
    }

    // CASE 1 — Existing player rejoining (same UUID preserved)
    if (existingPlayerId) {
        const teamId    = existingPlayerData[2]
        const isManager = existingPlayerData[3] === 'true'
        const isBot     = existingPlayerData[5]

        // status stays "online" here — this represents an active rejoin attempt
        // happening right now, immediately followed by a socket connection
        const updatedValue = [
            cleanNickname,
            playerPinHash,
            teamId,
            String(isManager),
            'offline',
            isBot
        ].join(':')

        const pipeline = redis.pipeline()

        pipeline.hset(`room:${roomId}:players`, { [existingPlayerId]: updatedValue })
        pipeline.expire(`room:${roomId}:players`, FOUR_DAYS_IN_SECONDS)

        pipeline.hset(`session:${existingPlayerId}`, {
            playerId:    existingPlayerId,
            roomId,
            teamId,
            nickname:    cleanNickname,
            isManager:   String(isManager),
            userPinHash: playerPinHash
        })
        pipeline.expire(`session:${existingPlayerId}`, FOUR_DAYS_IN_SECONDS)

        await pipeline.exec()

        return res.status(200).json(
            new ApiResponse(200, {
                playerId:         existingPlayerId,
                isExistingPlayer: true,
                teamId,
                isManager,
                roomStatus:       room.status
            }, "Rejoined room successfully")
        )
    }

    // CASE 2 — New player joining
    if (room.status !== 'lobby') {
        throw new ApiError(400, "Auction has already started. New players cannot join.")
    }

    const currentPlayerCount = await redis.hlen(`room:${roomId}:players`)
    if (currentPlayerCount >= MAX_PLAYERS_PER_ROOM) {
        throw new ApiError(400, "Room is full. Maximum 10 players allowed.")
    }

    const playerId = uuidv4()

    const pipeline = redis.pipeline()

    // status is now "offline" — only onReconnect (actual socket connection) sets "online"
    pipeline.hset(`room:${roomId}:players`, {
        [playerId]: `${cleanNickname}:${playerPinHash}::false:offline:false`
    })
    pipeline.expire(`room:${roomId}:players`, FOUR_DAYS_IN_SECONDS)

    pipeline.hset(`session:${playerId}`, {
        playerId,
        roomId,
        teamId:      '',
        nickname:    cleanNickname,
        isManager:   'false',
        userPinHash: playerPinHash
    })
    pipeline.expire(`session:${playerId}`, FOUR_DAYS_IN_SECONDS)

    await pipeline.exec()

    return res.status(200).json(
        new ApiResponse(200, {
            playerId,
            isExistingPlayer: false,
            teamId:           '',
            isManager:        false,
            roomStatus:       room.status
        }, "Joined room successfully")
    )
})

export { joinRoom }