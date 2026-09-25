// apps/web/src/lib/bidIncrement.js
import { nextBidAmount } from '@ipl-auction/shared'

/**
 * What the user's next bid WILL be, shown before they click. The server is
 * the source of truth; this uses the same shared rule it is tested against
 * (packages/shared/src/rules.js ↔ the placeBidAtomic Lua script).
 *
 * @param {number} currentBid - current bid amount (in lakhs), equals basePrice if no bids yet
 * @param {string} currentBidderId - teamId of current highest bidder, '' if no bids yet
 * @returns {number} the amount a NEW bid would be set to
 */
export const getNextBidAmount = (currentBid, currentBidderId) =>
  nextBidAmount(currentBid, Boolean(currentBidderId))
