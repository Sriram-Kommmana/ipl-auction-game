import { useMemo } from 'react'
import { useSessionStore } from '../store/sessionStore'
import { useRoomStore } from '../store/roomStore'
import { useAuctionStore } from '../store/auctionStore'
import { useSocketConnected } from './useSocketConnected'
import { getNextBidAmount } from '../lib/bidIncrement'

/**
 * Computes whether the current player's bid button should be disabled,
 * and why — based on the 5 rules:
 *   1. timerState !== 'RUNNING'
 *   2. myTeamId === currentBidderId (already the highest bidder)
 *   3. myTeam.purseLeft < nextBidAmount
 *   4. myTeam.playerCount >= maxPlayers
 *   5. nationality === 'Overseas' AND myTeam.overseasCount >= maxOverseas
 *
 * Also returns nextBidAmount so BidButton.jsx can label itself
 * (e.g. "Bid ₹60L") without recalculating it separately.
 */
export const useBidButton = () => {
  const myTeamId = useSessionStore((s) => s.teamId)

  const teams = useRoomStore((s) => s.teams)
  const maxPlayers = useRoomStore((s) => s.maxPlayers)
  const maxOverseas = useRoomStore((s) => s.maxOverseas)

  const timerState = useAuctionStore((s) => s.timerState)
  const currentBid = useAuctionStore((s) => s.currentBid)
  const currentBidderId = useAuctionStore((s) => s.currentBidderId)
  const currentPlayer = useAuctionStore((s) => s.currentPlayer)

  const isConnected = useSocketConnected()

  return useMemo(() => {
    const nextBidAmount = getNextBidAmount(currentBid, currentBidderId)
    const myTeam = teams.find((t) => t.teamId === myTeamId)

    // Checked first — if disconnected, nothing else matters
    if (!isConnected) {
      return { disabled: true, nextBidAmount, reason: 'disconnected' }
    }

    // Spectating (no team picked yet) or no active lot to bid on
    if (!myTeam || !currentPlayer) {
      return { disabled: true, nextBidAmount, reason: 'noTeamOrPlayer' }
    }

    if (timerState !== 'RUNNING') {
      return { disabled: true, nextBidAmount, reason: 'timerNotRunning' }
    }

    if (myTeamId === currentBidderId) {
      return { disabled: true, nextBidAmount, reason: 'alreadyHighestBidder' }
    }

    if (myTeam.purseLeft < nextBidAmount) {
      return { disabled: true, nextBidAmount, reason: 'insufficientPurse' }
    }

    if (myTeam.playerCount >= maxPlayers) {
      return { disabled: true, nextBidAmount, reason: 'squadFull' }
    }

    if (currentPlayer.nationality === 'Overseas' && myTeam.overseasCount >= maxOverseas) {
      return { disabled: true, nextBidAmount, reason: 'overseasLimitReached' }
    }

    return { disabled: false, nextBidAmount, reason: null }
  }, [myTeamId, teams, maxPlayers, maxOverseas, timerState, currentBid, currentBidderId, currentPlayer, isConnected])
}