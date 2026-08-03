import { useEffect, useState } from 'react'
import { useBidButton } from '../../hooks/useBidButton'
import { useSessionStore } from '../../store/sessionStore'
import { useAuctionStore } from '../../store/auctionStore'
import socket from '../../lib/socket'

const REASON_LABELS = {
  disconnected: 'Reconnecting…',
  noTeamOrPlayer: 'No active lot',
  timerNotRunning: 'Bidding closed',
  alreadyHighestBidder: "You're the highest bidder",
  insufficientPurse: 'Not enough purse',
  squadFull: 'Squad full',
  overseasLimitReached: 'Overseas limit reached'
}

const BidButton = () => {
  const { disabled, nextBidAmount, reason } = useBidButton()
  const playerId = useSessionStore((s) => s.playerId)
  const currentBid = useAuctionStore((s) => s.currentBid)
  const currentBidderId = useAuctionStore((s) => s.currentBidderId)

  const [isPending, setIsPending] = useState(false)

  // Any change to bid state means the server processed SOME outcome for
  // this lot (ours or someone else's) — safe to re-enable immediately,
  // rather than waiting out the full safety timeout below.
  useEffect(() => {
    setIsPending(false)
  }, [currentBid, currentBidderId])

  // Safety net: if nothing comes back within a few seconds (dropped
  // packet, brief disconnect), don't leave the button stuck disabled.
  useEffect(() => {
    if (!isPending) return
    const timeout = setTimeout(() => setIsPending(false), 3000)
    return () => clearTimeout(timeout)
  }, [isPending])

  const handleBid = () => {
    if (isPending) return
    setIsPending(true)
    socket.emit('placeBid', { playerId })
  }

  const isDisabled = disabled || isPending

  return (
    <div className="text-center">
      <button
        type="button"
        onClick={handleBid}
        disabled={isDisabled}
        className="w-full bg-brand-red hover:bg-brand-red-dark disabled:opacity-40 disabled:cursor-not-allowed
                   text-paper font-display text-3xl tracking-wide py-5 rounded-2xl transition-colors"
      >
        {isPending ? 'BIDDING…' : `BID ₹${nextBidAmount}L`}
      </button>
      {isDisabled && !isPending && reason && REASON_LABELS[reason] && (
        <p className="text-xs text-ink/40 mt-2">{REASON_LABELS[reason]}</p>
      )}
    </div>
  )
}

export default BidButton