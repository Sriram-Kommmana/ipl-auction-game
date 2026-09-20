// apps/web/src/components/lobby/LobbyControls.jsx
import { useState } from 'react'
import { useSessionStore } from '../../store/sessionStore'
import { useRoomStore } from '../../store/roomStore'
import { useSocketConnected } from '../../hooks/useSocketConnected'
import { useLeaveRoom } from '../../hooks/useLeaveRoom'
import Modal from '../shared/Modal'
import socket from '../../lib/socket'

const LobbyControls = () => {
  const isManager = useSessionStore((s) => s.isManager)
  const playerId = useSessionStore((s) => s.playerId)
  const teams = useRoomStore((s) => s.teams)
  const isConnected = useSocketConnected()
  const leaveRoom = useLeaveRoom()

  const [showLeaveConfirm, setShowLeaveConfirm] = useState(false)

  const claimedCount = teams.filter((t) => t.ownerId).length
  const isStartDisabled = claimedCount < 2 || !isConnected

  const handleStartAuction = () => {
    socket.emit('startAuction', { playerId })
  }

  return (
    <>
      <div className="mt-8 pt-6 border-t border-line flex flex-col-reverse items-stretch gap-5 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
        <button
          type="button"
          onClick={() => setShowLeaveConfirm(true)}
          className="link-back self-center sm:self-auto"
        >
          ← Leave Room
        </button>

        {isManager && (
          <div className="text-center sm:text-right">
            <button
              type="button"
              onClick={handleStartAuction}
              disabled={isStartDisabled}
              className="btn-primary w-full sm:w-auto text-2xl px-8 py-3"
            >
              {!isConnected ? 'Reconnecting…' : 'Start Auction →'}
            </button>
            {isConnected && claimedCount < 2 && (
              <p className="font-mono text-[10px] uppercase tracking-wider text-bone/40 mt-3">
                <span className="text-red">{claimedCount}/2</span> · Need at least 2 teams claimed
              </p>
            )}
          </div>
        )}
      </div>

      <Modal
        isOpen={showLeaveConfirm}
        title="Leave this room?"
        message="You'll be able to rejoin from the Home screen's Continue Playing list."
        confirmLabel="Leave"
        danger
        onConfirm={leaveRoom}
        onCancel={() => setShowLeaveConfirm(false)}
      />
    </>
  )
}

export default LobbyControls