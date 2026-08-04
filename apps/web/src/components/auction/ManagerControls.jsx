// apps/web/src/components/auction/ManagerControls.jsx
import { useEffect, useState } from 'react'
import { useSessionStore } from '../../store/sessionStore'
import { useAuctionStore } from '../../store/auctionStore'
import { useSocketConnected } from '../../hooks/useSocketConnected'
import { useLeaveRoom } from '../../hooks/useLeaveRoom'
import socket from '../../lib/socket'

const btnClass =
  'font-display text-sm tracking-wide px-4 py-2 rounded-lg transition-colors ' +
  'disabled:opacity-40 disabled:cursor-not-allowed'

const ManagerControls = () => {
  const isManager = useSessionStore((s) => s.isManager)
  const playerId = useSessionStore((s) => s.playerId)
  const timerState = useAuctionStore((s) => s.timerState)
  const currentPlayerIndex = useAuctionStore((s) => s.currentPlayerIndex)
  const isConnected = useSocketConnected()
  const leaveRoom = useLeaveRoom()

  const [isSkipping, setIsSkipping] = useState(false)

  // Deliberately only currentPlayerIndex — NOT timerState. Skip itself
  // flips timerState to PROCESSING_EXPIRY immediately as part of the same
  // action (acquireExpiryLock), before nextPlayer arrives. Resetting on
  // timerState change would flash the button clickable again during that
  // in-between window. The 3s safety timeout below already covers the
  // rare edge case (network glitch, index doesn't change) this would have
  // been trying to solve.
  useEffect(() => {
    setIsSkipping(false)
  }, [currentPlayerIndex])

  useEffect(() => {
    if (!isSkipping) return
    const timeout = setTimeout(() => setIsSkipping(false), 3000)
    return () => clearTimeout(timeout)
  }, [isSkipping])

  const handlePause = () => {
    if (!isConnected) return
    socket.emit('pauseAuction', { playerId })
  }

  const handleResume = () => {
    if (!isConnected) return
    socket.emit('resumeAuction', { playerId })
  }

  const handleSkip = () => {
    if (!isConnected || isSkipping) return
    setIsSkipping(true)
    socket.emit('skipPlayer', { playerId })
  }

  return (
    <div className="mt-6 flex items-center justify-between">
      <button
        type="button"
        onClick={leaveRoom}
        className="text-sm text-ink/50 hover:text-brand-red transition-colors"
      >
        ← Leave Room
      </button>

      {isManager && (
        <div className="flex gap-2">
          {timerState === 'RUNNING' && (
            <button
              type="button"
              onClick={handlePause}
              disabled={!isConnected}
              className={`${btnClass} bg-ink text-paper hover:bg-charcoal`}
            >
              PAUSE
            </button>
          )}

          {timerState === 'PAUSED' && (
            <button
              type="button"
              onClick={handleResume}
              disabled={!isConnected}
              className={`${btnClass} bg-ink text-paper hover:bg-charcoal`}
            >
              RESUME
            </button>
          )}

          {timerState === 'RUNNING' && (
            <button
              type="button"
              onClick={handleSkip}
              disabled={!isConnected || isSkipping}
              className={`${btnClass} bg-mist text-ink border border-line hover:border-brand-red`}
            >
              {isSkipping ? 'SKIPPING…' : 'SKIP'}
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export default ManagerControls