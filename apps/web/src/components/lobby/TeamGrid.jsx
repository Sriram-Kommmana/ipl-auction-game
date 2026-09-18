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
    <div className="panel p-4">
      <div className="section-head">
        <span className="section-num">01</span>
        <h2 className="section-title">Choose Your Team</h2>
        <span className="section-jp">チーム選択</span>
      </div>
      {!isConnected && (
        <p className="font-mono text-xs text-red border-l-2 border-red bg-red/10 px-3 py-2 mb-3">
          ! Reconnecting… team selection is paused.
        </p>
      )}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
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
              className={`group relative overflow-hidden text-left border bg-void transition-all min-h-[118px] flex flex-col
                ${isMine
                  ? 'border-cyan shadow-[4px_4px_0_0_var(--color-cyan)] -translate-x-0.5 -translate-y-0.5'
                  : 'border-line-strong'}
                ${isDisabled
                  ? 'opacity-35 grayscale-[60%] cursor-not-allowed'
                  : isMine ? '' : 'hover:border-bone hover:-translate-x-0.5 hover:-translate-y-0.5 hover:shadow-[4px_4px_0_0_var(--color-red)] cursor-pointer'}
              `}
              style={{
                backgroundImage: `linear-gradient(160deg, ${team.color}40 0%, transparent 65%)`
              }}
            >
              {/* Brand stripe: primary + accent */}
              <div className="flex h-1.5 shrink-0">
                <div className="flex-[3]" style={{ backgroundColor: team.color }} />
                <div className="flex-1" style={{ backgroundColor: team.accent }} />
              </div>

              <div className="p-3 flex flex-col flex-1">
                <p className="font-display text-3xl leading-none text-bone tracking-wide">
                  {team.teamId}
                </p>
                <p className="text-[11px] leading-tight text-bone/55 mt-1 line-clamp-2">
                  {team.name}
                </p>
                <div className="mt-auto pt-2">
                  {claim?.ownerId ? (
                    <p className={`font-mono text-[10px] uppercase tracking-wider truncate
                      ${isMine ? 'text-cyan' : 'text-bone/70'}`}
                    >
                      {isMine ? '● Your Team' : `◆ ${nicknameByPlayerId[claim.ownerId] ?? 'Loading...'}`}
                    </p>
                  ) : (
                    <p className="font-mono text-[10px] uppercase tracking-wider text-bone/25 group-hover:text-red transition-colors">
                      ○ Open
                    </p>
                  )}
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

export default TeamGrid