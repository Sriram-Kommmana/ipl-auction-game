import { TEAMS } from '../../constants/teams'
import { useRoomStore } from '../../store/roomStore'
import { useSessionStore } from '../../store/sessionStore'
import { useSocketConnected } from '../../hooks/useSocketConnected'
import socket from '../../lib/socket'

const TeamGrid = () => {
  const teams = useRoomStore((s) => s.teams)
  const players = useRoomStore((s) => s.players)
  const myPlayerId = useSessionStore((s) => s.playerId)
  const myTeamId = useSessionStore((s) => s.teamId)
  const isConnected = useSocketConnected()

  const claimedByTeamId = Object.fromEntries(teams.map((t) => [t.teamId, t]))
  const nicknameByPlayerId = Object.fromEntries(players.map((p) => [p.playerId, p.nickname]))

  const handleSelect = (teamId) => {
    if (teamId === myTeamId) return
    socket.emit('selectTeam', { playerId: myPlayerId, teamId })
  }

  return (
    <div>
      <h2 className="font-display text-2xl text-ink mb-3 tracking-wide">CHOOSE YOUR TEAM</h2>
      {!isConnected && (
        <p className="text-xs text-brand-red-dark mb-2">Reconnecting… team selection is paused.</p>
      )}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {TEAMS.map((team) => {
          const claim = claimedByTeamId[team.teamId]
          const isMine = team.teamId === myTeamId
          // Check claim?.ownerId, NOT just whether an entry exists — switching
          // teams clears ownerId to null but leaves the array entry in place
          // (see roomStore.applyTeamSelection's clearPrevious), so "entry
          // exists" alone would wrongly treat an unclaimed team as taken.
          const isTakenByOther = !!claim?.ownerId && !isMine
          const isDisabled = isTakenByOther || !isConnected

          return (
            <button
              key={team.teamId}
              type="button"
              disabled={isDisabled}
              onClick={() => handleSelect(team.teamId)}
              className={`relative rounded-xl p-4 text-left border-2 transition-all
                ${isMine ? 'border-ink shadow-lg' : 'border-transparent'}
                ${isDisabled ? 'opacity-40 cursor-not-allowed' : 'hover:scale-[1.03] cursor-pointer'}
              `}
              style={{ backgroundColor: team.color }}
            >
              <p className="font-display text-lg text-white drop-shadow leading-tight truncate">
                {team.name}
              </p>
              <p className="text-xs text-white/70 mt-0.5 tracking-widest">{team.teamId}</p>
              {claim?.ownerId && (
                <p className="text-xs text-white mt-2 font-semibold truncate">
                  {isMine ? 'YOUR TEAM' : (nicknameByPlayerId[claim.ownerId] ?? 'Loading...')}
                </p>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}

export default TeamGrid