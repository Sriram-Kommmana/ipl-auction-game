// apps/web/src/components/lobby/LobbyControls.jsx
import { useSessionStore } from '../../store/sessionStore'
import { useRoomStore } from '../../store/roomStore'
import { useSocketConnected } from '../../hooks/useSocketConnected'
import { useLeaveRoom } from '../../hooks/useLeaveRoom'
import socket from '../../lib/socket'

const LobbyControls = ({ socketError }) => {
  const isManager = useSessionStore((s) => s.isManager)
  const playerId = useSessionStore((s) => s.playerId)
  const teams = useRoomStore((s) => s.teams)
  const isConnected = useSocketConnected()
  const leaveRoom = useLeaveRoom()

  const claimedCount = teams.filter((t) => t.ownerId).length
  const isStartDisabled = claimedCount < 2 || !isConnected

  const handleStartAuction = () => {
    socket.emit('startAuction', { playerId })
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
        <div className="text-right">
          {socketError && (
            <p className="text-sm text-brand-red-dark mb-2">{socketError.message}</p>
          )}
          <button
            type="button"
            onClick={handleStartAuction}
            disabled={isStartDisabled}
            className="bg-brand-red hover:bg-brand-red-dark disabled:opacity-40 disabled:cursor-not-allowed
                       text-paper font-display text-xl tracking-wide px-6 py-3 rounded-lg transition-colors"
          >
            {!isConnected ? 'RECONNECTING…' : 'START AUCTION'}
          </button>
          {isConnected && claimedCount < 2 && (
            <p className="text-xs text-ink/40 mt-1">Need at least 2 teams claimed</p>
          )}
        </div>
      )}
    </div>
  )
}

export default LobbyControls