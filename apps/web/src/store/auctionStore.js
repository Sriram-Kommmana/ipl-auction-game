import { create } from 'zustand'

export const useAuctionStore = create((set) => ({
  // ---- state ----
  currentPlayerIndex: 0,
  currentPlayer: null,        // { slNo, playerName, country, nationality, role, basePrice, rating, stats }
  currentBid: 0,
  currentBidderId: '',        // '' = no bids yet, otherwise a teamId
  timerState: 'IDLE',         // IDLE | RUNNING | PAUSED | PROCESSING_EXPIRY | ENDED
  timerEndsAt: null,          // unix timestamp (seconds) — NOT a live countdown, see useTimer.js
  pausedTimeRemaining: null,  // seconds left, only set while PAUSED
  isReauction: false,
  bidFeed: [],                // recent bids for the CURRENT player only — resets every lot

  // Transient result shown between playerSold/playerUnsold and nextPlayer —
  // e.g. a "SOLD to RCB for 2200L!" banner. Backend fires these events
  // back-to-back with no gap, so useSocket.js is responsible for holding
  // this on screen briefly (setTimeout) before nextPlayer clears it.
  lastResult: null,           // { status: 'sold'|'unsold', playerName, soldTo, teamName, soldFor } | null

  // ---- actions ----

  // Used on stateSync — full reconnect snapshot
  setAuctionState: (auctionData) => set({
    currentPlayerIndex: auctionData.currentPlayerIndex,
    currentPlayer: auctionData.currentPlayer,
    currentBid: auctionData.currentBid,
    currentBidderId: auctionData.currentBidderId,
    timerState: auctionData.timerState,
    timerEndsAt: auctionData.timerEndsAt,
    pausedTimeRemaining: auctionData.pausedTimeRemaining,
    isReauction: auctionData.isReauction || false
  }),

  // Used on auctionStarted — the very first player of the auction.
  // isReauction explicitly false — starting the auction always begins main phase.
  startAuction: ({ currentPlayerIndex, player, currentBid, currentBidderId }) => set({
    currentPlayerIndex,
    currentPlayer: player,
    currentBid,
    currentBidderId,
    isReauction: false,
    bidFeed: [],
    lastResult: null
  }),

  // Used on nextPlayer — advancing to a new lot resets bid-specific fields
  // and clears the bid feed, since it only ever shows the CURRENT player's bids.
  //
  // isReauction is STICKY: the backend only sends `isReauction: true` on the
  // FIRST player of a re-auction round — every subsequent nextPlayer in that
  // same round omits the flag entirely. So we only update it when the event
  // explicitly includes it; otherwise we keep whatever it already was. This
  // avoids incorrectly flipping back to false for reauction players 2, 3, etc.
  goToNextPlayer: ({ currentPlayerIndex, player, currentBid, currentBidderId, isReauction }) => set((state) => ({
    currentPlayerIndex,
    currentPlayer: player,
    currentBid,
    currentBidderId,
    isReauction: isReauction !== undefined ? isReauction : state.isReauction,
    bidFeed: [],
    lastResult: null
  })),

  // Used on bidPlaced
  applyBid: (teamId, newBid) => set((state) => ({
    currentBid: newBid,
    currentBidderId: teamId,
    bidFeed: [...state.bidFeed, { teamId, bid: newBid }]
  })),

  // Used on timerStarted / timerResumed (same shape, both just (re)start the clock)
  setTimerRunning: (timerEndsAt) => set({
    timerState: 'RUNNING',
    timerEndsAt,
    pausedTimeRemaining: null
  }),

  // Used on timerPaused
  setTimerPaused: (pausedTimeRemaining) => set({
    timerState: 'PAUSED',
    timerEndsAt: null,
    pausedTimeRemaining
  }),

  // Used on playerSold / playerUnsold — populates the transient banner.
  // useSocket.js clears this itself (setTimeout) or nextPlayer clears it
  // automatically via goToNextPlayer above.
  setLastResult: (result) => set({ lastResult: result }),
  clearLastResult: () => set({ lastResult: null }),

  // Used when leaving a room — same reasoning as roomStore.resetRoom
  resetAuction: () => set({
    currentPlayerIndex: 0,
    currentPlayer: null,
    currentBid: 0,
    currentBidderId: '',
    timerState: 'IDLE',
    timerEndsAt: null,
    pausedTimeRemaining: null,
    isReauction: false,
    bidFeed: [],
    lastResult: null
  })
}))