import { useState, useEffect } from 'react'
import { useAuctionStore } from '../store/auctionStore'

/**
 * Turns auctionStore's static timerEndsAt (a unix timestamp, seconds since
 * epoch) into a live, ticking secondsLeft value — entirely in LOCAL state,
 * never written into Zustand. If this lived in a store, every subscribed
 * component would re-render every single second during the whole auction.
 */
export const useTimer = () => {
  const timerState = useAuctionStore((s) => s.timerState)
  const timerEndsAt = useAuctionStore((s) => s.timerEndsAt)
  const pausedTimeRemaining = useAuctionStore((s) => s.pausedTimeRemaining)

  const [secondsLeft, setSecondsLeft] = useState(0)

  useEffect(() => {
    // Paused — show the frozen value, no ticking interval needed
    if (timerState === 'PAUSED') {
      setSecondsLeft(pausedTimeRemaining ?? 0)
      return
    }

    // Nothing running (IDLE, PROCESSING_EXPIRY, ENDED, or no timestamp yet)
    if (timerState !== 'RUNNING' || !timerEndsAt) {
      setSecondsLeft(0)
      return
    }

    const tick = () => {
      const remaining = timerEndsAt - Math.floor(Date.now() / 1000)
      setSecondsLeft(Math.max(remaining, 0))
    }

    tick() // run immediately — don't wait 1s for the first paint
    const intervalId = setInterval(tick, 1000)

    return () => clearInterval(intervalId)
  }, [timerState, timerEndsAt, pausedTimeRemaining])

  return { secondsLeft, timerState }
}