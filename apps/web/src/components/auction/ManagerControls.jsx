// apps/web/src/components/auction/ManagerControls.jsx
// NOTE: "Leave Room" moved OUT of this component — it's now in Auction.jsx's
// header row (available to every player, not just the manager). This
// component now only renders the manager-only Pause/Resume/Skip group.
import { useEffect, useState } from 'react'
import { useSessionStore } from '../../store/sessionStore'
import { useAuctionStore } from '../../store/auctionStore'
import { useSocketConnected } from '../../hooks/useSocketConnected'
import { TEAMS_BY_ID } from '../../constants/teams'
import Modal from '../shared/Modal'
import socket from '../../lib/socket'

const btnClass =
  'font-display text-sm tracking-[0.08em] whitespace-nowrap px-3 sm:px-4 py-1.5 border-2 transition-colors ' +
  'disabled:opacity-40 disabled:cursor-not-allowed'

const ManagerControls = () => {
  const isManager = useSessionStore((s) => s.isManager)
  const playerId = useSessionStore((s) => s.playerId)
  const timerState = useAuctionStore((s) => s.timerState)
  const currentPlayerIndex = useAuctionStore((s) => s.currentPlayerIndex)
  const currentBid = useAuctionStore((s) => s.currentBid)
  const currentBidderId = useAuctionStore((s) => s.currentBidderId)
  const isConnected = useSocketConnected()

  const [isSkipping, setIsSkipping] = useState(false)
  const [showSkipConfirm, setShowSkipConfirm] = useState(false)

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

  const doSkip = () => {
    if (!isConnected || isSkipping) return
    setIsSkipping(true)
    socket.emit('skipPlayer', { playerId })
  }

  const handleSkipClick = () => {
    if (!isConnected || isSkipping) return
    if (currentBidderId) {
      setShowSkipConfirm(true)
    } else {
      doSkip()
    }
  }

  const confirmSkip = () => {
    setShowSkipConfirm(false)
    doSkip()
  }

  const bidderTeam = currentBidderId ? TEAMS_BY_ID[currentBidderId] : null

  if (!isManager) return null

  return (
    <>
      <div className="flex gap-2">
        {timerState === 'RUNNING' && (
          <button
            type="button"
            onClick={handlePause}
            disabled={!isConnected}
            className={`${btnClass} border-amber text-amber hover:bg-amber hover:text-void`}
          >
            <span className="hidden sm:inline">❚❚ </span>PAUSE
          </button>
        )}

        {timerState === 'PAUSED' && (
          <button
            type="button"
            onClick={handleResume}
            disabled={!isConnected}
            className={`${btnClass} border-cyan text-cyan hover:bg-cyan hover:text-void`}
          >
            <span className="hidden sm:inline">▶ </span>RESUME
          </button>
        )}

        {timerState === 'RUNNING' && (
          <button
            type="button"
            onClick={handleSkipClick}
            disabled={!isConnected || isSkipping}
            className={`${btnClass} border-line-strong text-bone/80 hover:border-red hover:text-red`}
          >
            {isSkipping ? 'SKIPPING…' : <>SKIP<span className="hidden sm:inline"> ▸▸</span></>}
          </button>
        )}
      </div>

      <Modal
        isOpen={showSkipConfirm}
        title="Skip with an active bid?"
        message={
          bidderTeam
            ? `${bidderTeam.teamId} currently has the highest bid at ₹${currentBid}L. Skipping will discard this bid — nobody gets the player, no purse is charged.`
            : 'This will discard the current bid entirely.'
        }
        confirmLabel="Skip Anyway"
        danger
        onConfirm={confirmSkip}
        onCancel={() => setShowSkipConfirm(false)}
      />
    </>
  )
}

export default ManagerControls