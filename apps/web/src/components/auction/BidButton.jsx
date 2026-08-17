// apps/web/src/components/auction/BidButton.jsx
import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useBidButton } from '../../hooks/useBidButton'
import { useTimer } from '../../hooks/useTimer'
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
  const { secondsLeft, timerState, totalDuration } = useTimer()

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

  // Only show the depleting bar while actually running — no bar during
  // IDLE/PAUSED/PROCESSING_EXPIRY, since "time remaining" isn't meaningful then.
  const progressPercent = timerState === 'RUNNING' && totalDuration > 0
    ? Math.max(0, Math.min(100, (secondsLeft / totalDuration) * 100))
    : 0

  return (
    <div className="text-center h-[88px]">
      <motion.button
        type="button"
        onClick={handleBid}
        disabled={isDisabled}
        whileTap={isDisabled ? {} : { scale: 0.96 }}
        className="relative w-full h-16 bg-brand-red hover:bg-brand-red-dark disabled:opacity-40 disabled:cursor-not-allowed
           text-paper font-display text-3xl tracking-wide rounded-2xl transition-colors overflow-hidden"
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
            className="inline-block relative z-10"
          >
            {isPending ? 'BIDDING…' : `BID ₹${nextBidAmount}L`}
          </motion.span>
        </AnimatePresence>

        {/* Depleting progress bar — width shrinks as secondsLeft counts
            down toward 0. transition duration matches the 1s tick from
            useTimer so it animates smoothly rather than jumping. */}
        {timerState === 'RUNNING' && (
          <motion.div
            className="absolute left-0 bottom-0 h-1 bg-paper/60"
            animate={{ width: `${progressPercent}%` }}
            transition={{ duration: 1, ease: 'linear' }}
          />
        )}
      </motion.button>
      {isDisabled && !isPending && reason && REASON_LABELS[reason] && (
        <p className="text-xs text-ink/40 mt-2">{REASON_LABELS[reason]}</p>
      )}
    </div>
  )
}

export default BidButton