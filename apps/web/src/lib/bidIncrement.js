// apps/web/src/lib/bidIncrement.js

/**
 * Mirrors placeBidAtomic's Lua logic exactly — used to show the user
 * what their next bid amount WILL be, before they click.
 * The server is still the source of truth; this is purely a UI preview.
 *
 * @param {number} currentBid - current bid amount (in lakhs), equals basePrice if no bids yet
 * @param {string} currentBidderId - teamId of current highest bidder, '' if no bids yet
 * @returns {number} the amount a NEW bid would be set to
 */
export const getNextBidAmount = (currentBid, currentBidderId) => {
  // First bid on this player — no increment, just the base price
  if (!currentBidderId) {
    return currentBid
  }

  let increment
  if (currentBid < 200) {
    increment = 10
  } else if (currentBid < 300) {
    increment = 20
  } else {
    increment = 50
  }

  return currentBid + increment
}