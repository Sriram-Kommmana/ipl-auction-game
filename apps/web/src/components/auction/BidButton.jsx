import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
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

  useEffect(() => {
    setIsPending(false)
  }, [currentBid, currentBidderId])

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
      <motion.button
        type="button"
        onClick={handleBid}
        disabled={isDisabled}
        whileTap={isDisabled ? {} : { scale: 0.96 }}
        className="w-full bg-brand-red hover:bg-brand-red-dark disabled:opacity-40 disabled:cursor-not-allowed
                   text-paper font-display text-3xl tracking-wide py-5 rounded-2xl transition-colors overflow-hidden"
      >
        {/* Keyed by the label content itself — pulses whenever the target
            bid amount changes (or pending state toggles), signaling
            "the price you'd pay just moved, look here". */}
        <AnimatePresence mode="wait">
          <motion.span
            key={isPending ? 'pending' : nextBidAmount}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.15 }}
            className="inline-block"
          >
            {isPending ? 'BIDDING…' : `BID ₹${nextBidAmount}L`}
          </motion.span>
        </AnimatePresence>
      </motion.button>
      {isDisabled && !isPending && reason && REASON_LABELS[reason] && (
        <p className="text-xs text-ink/40 mt-2">{REASON_LABELS[reason]}</p>
      )}
    </div>
  )
}

export default BidButton