import { ApiError } from '../utils/ApiError.js'
import { ApiResponse } from '../utils/ApiResponse.js'
import { asyncHandler } from '../utils/asyncHandler.js'
import AuctionResult from '../db/models/AuctionResult.js'

const getResults = asyncHandler(async (req, res) => {
    const { roomId } = req.params

    if (!roomId?.trim()) {
        throw new ApiError(400, 'Room ID is required')
    }

    const result = await AuctionResult.findOne({ roomId }).lean()

    if (!result) {
        // persistAuctionResults is fire-and-forget after auctionCompleted —
        // the frontend may legitimately arrive here before MongoDB has
        // finished writing. This message is worded for that case; the
        // frontend retries on 404 rather than treating it as a hard failure.
        throw new ApiError(404, 'Results not ready yet. Please try again shortly.')
    }

    return res.status(200).json(
        new ApiResponse(200, result, 'Auction results retrieved successfully')
    )
})

export { getResults }