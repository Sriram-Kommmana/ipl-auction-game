// apps/server/src/http/getStats.js
//
// Public, read-only counters for the landing page ("ROOMS CREATED", etc).
// Purely additive: it reads nothing but document counts and changes no
// auction logic or state.
//
// Room documents are never deleted on expiry (only rolled back if the
// Redis write fails during creation), so countDocuments() on `rooms` is a
// true lifetime "rooms created" figure.
import { ApiResponse } from '../utils/ApiResponse.js'
import { asyncHandler } from '../utils/asyncHandler.js'
import Room from '../db/models/Room.js'
import Player from '../db/models/Player.js'

// The landing page is static and cached by browsers/CDNs anyway — a short
// server-side cache keeps repeat visits off the database entirely.
const CACHE_TTL_MS = 60 * 1000
let cache = { at: 0, data: null }

const getStats = asyncHandler(async (req, res) => {
    // Public aggregate numbers, no credentials — safe for any origin, so
    // the static landing page can read them without touching the app's
    // own CORS configuration.
    res.set('Access-Control-Allow-Origin', '*')
    res.set('Cache-Control', 'public, max-age=60')

    if (cache.data && Date.now() - cache.at < CACHE_TTL_MS) {
        return res.status(200).json(new ApiResponse(200, cache.data, 'Stats fetched (cached)'))
    }

    const [roomsCreated, playersAvailable] = await Promise.all([
        Room.estimatedDocumentCount(),
        Player.countDocuments({ isActive: true })
    ])

    const data = { roomsCreated, playersAvailable }
    cache = { at: Date.now(), data }

    return res.status(200).json(new ApiResponse(200, data, 'Stats fetched successfully'))
})

export { getStats }
