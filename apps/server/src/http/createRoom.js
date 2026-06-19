import { ApiError } from '../utils/ApiError.js'
import { ApiResponse } from '../utils/ApiResponse.js'
import { asyncHandler } from "../utils/asyncHandler.js";
import { customAlphabet } from 'nanoid'
import crypto from 'crypto'
import { v4 as uuidv4 } from 'uuid'
import redis from '../redis/client.js'
import Room from '../db/models/Room.js'
import Player from '../db/models/Player.js'

const generateRoomId = customAlphabet('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 6)

const FOUR_DAYS_IN_SECONDS = 4 * 24 * 60 * 60

const hashPin = (pin) => crypto.createHash('sha256').update(pin).digest('hex')

const shuffleArray = (array) => {
    const arr = [...array]
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]]
    }
    return arr
}

const buildAuctionPool = (players) => {
    const sets = {}
    for (const player of players) {
        if (!sets[player.setNo]) sets[player.setNo] = []
        sets[player.setNo].push(player.slNo)
    }

    const sortedSetKeys = Object.keys(sets).sort((a, b) => Number(a) - Number(b))
    const pool = []
    for (const key of sortedSetKeys) {
        const shuffled = shuffleArray(sets[key])
        pool.push(...shuffled)
    }

    return pool
}

const createRoom = asyncHandler(async (req, res) => {
    const {
        managerNickname,
        managerPin,
        roomPin,
        pursePerTeam = 12500
    } = req.body

    if (!managerNickname?.trim())
        throw new ApiError(400, "Manager nickname is required")
    if (managerNickname.trim().length < 2)
        throw new ApiError(400, "Manager nickname must be at least 2 characters")
    if (managerNickname.trim().length > 20)
        throw new ApiError(400, "Manager nickname must be at most 20 characters")
    if (managerNickname.trim().includes(':'))
        throw new ApiError(400, "Nickname cannot contain ':'")

    if (!managerPin)
        throw new ApiError(400, "Manager PIN is required")
    if (!/^\d{4}$/.test(String(managerPin)))
        throw new ApiError(400, "Manager PIN must be exactly 4 digits")

    if (!roomPin)
        throw new ApiError(400, "Room PIN is required")
    if (!/^\d{4}$/.test(String(roomPin)))
        throw new ApiError(400, "Room PIN must be exactly 4 digits")

    if (typeof pursePerTeam !== 'number' || pursePerTeam <= 0)
        throw new ApiError(400, "Purse per team must be a positive number")
    if (pursePerTeam < 5000)
        throw new ApiError(400, "Purse per team must be at least 5000 lakhs")
    if (pursePerTeam > 50000)
        throw new ApiError(400, "Purse per team cannot exceed 50000 lakhs")

    const players = await Player.find({ isActive: true }).sort({ setNo: 1 })
    if (!players || players.length === 0)
        throw new ApiError(500, "No players found. Please seed the database first.")

    const pool = buildAuctionPool(players)

    let roomId
    let exists = true
    while (exists) {
        roomId = generateRoomId()
        exists = await redis.exists(`room:${roomId}`)
    }

    const managerId       = uuidv4()
    const managerPinHash  = hashPin(String(managerPin))
    const roomPinHash     = hashPin(String(roomPin))

    const now       = Math.floor(Date.now() / 1000)
    const expiresAt = now + FOUR_DAYS_IN_SECONDS

    let roomRecord
    try {
        roomRecord = await Room.create({
            roomId,
            createdAt: new Date(),
            expiresAt: new Date(expiresAt * 1000)
        })
    } catch (err) {
        throw new ApiError(500, "Failed to create room record")
    }

    try {
        const pipeline = redis.pipeline()

        pipeline.hset(`room:${roomId}`, {
            roomId,
            managerPlayerId:     managerId,
            roomPinHash,
            status:              'lobby',
            auctionPhase:        'main',
            currentPlayerIndex:  0,
            pursePerTeam,
            timerDuration:       30,
            maxPlayers:          25,
            maxOverseas:         8,
            totalTeams:          0,
            createdAt:           now,
            startedAt:           '',
            completedAt:         ''
        })
        pipeline.expire(`room:${roomId}`, FOUR_DAYS_IN_SECONDS)

        pipeline.del(`room:${roomId}:teams`)
        pipeline.expire(`room:${roomId}:teams`, FOUR_DAYS_IN_SECONDS)

        pipeline.rpush(`room:${roomId}:pool`, ...pool.map(String))
        pipeline.expire(`room:${roomId}:pool`, FOUR_DAYS_IN_SECONDS)

        const poolStatusArgs = {}
        for (const slNo of pool) {
            poolStatusArgs[String(slNo)] = 'pending'
        }
        pipeline.hset(`room:${roomId}:pool:status`, poolStatusArgs)
        pipeline.expire(`room:${roomId}:pool:status`, FOUR_DAYS_IN_SECONDS)

        pipeline.del(`room:${roomId}:pool:unsold`)
        pipeline.expire(`room:${roomId}:pool:unsold`, FOUR_DAYS_IN_SECONDS)

        pipeline.del(`room:${roomId}:history`)
        pipeline.expire(`room:${roomId}:history`, FOUR_DAYS_IN_SECONDS)

        pipeline.del(`room:${roomId}:chat`)
        pipeline.expire(`room:${roomId}:chat`, FOUR_DAYS_IN_SECONDS)

        pipeline.hset(`room:${roomId}:players`, {
            [managerId]: `${managerNickname.trim()}:${managerPinHash}::true:online:false`
        })
        pipeline.expire(`room:${roomId}:players`, FOUR_DAYS_IN_SECONDS)

        pipeline.hset(`room:${roomId}:current`, {
            iplPlayerId:         '',
            basePrice:           '',
            currentBid:          '',
            currentBidderId:     '',
            timerState:          'IDLE',
            timerEndsAt:         '',
            pausedTimeRemaining: ''
        })
        pipeline.expire(`room:${roomId}:current`, FOUR_DAYS_IN_SECONDS)

        // Session key — includes isManager + userPinHash
        pipeline.hset(`session:${managerId}`, {
            playerId:    managerId,
            roomId,
            teamId:      '',
            nickname:    managerNickname.trim(),
            isManager:   'true',
            userPinHash: managerPinHash
        })
        pipeline.expire(`session:${managerId}`, FOUR_DAYS_IN_SECONDS)

        await pipeline.exec()

    } catch (err) {
        await Room.deleteOne({ _id: roomRecord._id })
        throw new ApiError(500, "Failed to initialize room. Please try again.")
    }

    return res.status(201).json(
        new ApiResponse(201, {
            roomId,
            managerId,
            teamId: '',
            isManager: true
        }, "Room created successfully")
    )
})

export { createRoom }