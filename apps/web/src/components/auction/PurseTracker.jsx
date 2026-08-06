import { useMemo } from 'react'
import { TEAMS_BY_ID } from '../../constants/teams'
import { useRoomStore } from '../../store/roomStore'
import { useSessionStore } from '../../store/sessionStore'

const PurseTracker = () => {
  const teams = useRoomStore((s) => s.teams)
  const players = useRoomStore((s) => s.players)
  const myTeamId = useSessionStore((s) => s.teamId)

  const nicknameByPlayerId = useMemo(
    () => Object.fromEntries(players.map((p) => [p.playerId, p.nickname])),
    [players]
  )

  const sorted = useMemo(() => {
    const claimed = teams.filter((t) => t.ownerId)
    return [...claimed].sort((a, b) => b.purseLeft - a.purseLeft)
  }, [teams])

  const topPurse = sorted[0]?.purseLeft

  if (sorted.length === 0) {
    return (
      <div className="bg-paper border border-line rounded-2xl p-4">
        <h2 className="font-display text-lg text-ink mb-2 tracking-wide">PURSE TRACKER</h2>
        <p className="text-xs text-ink/30 text-center py-4">No teams have joined yet</p>
      </div>
    )
  }

  return (
    <div className="bg-paper border border-line rounded-2xl p-4">
      <div className="flex items-center justify-between mb-2">
        <h2 className="font-display text-lg text-ink tracking-wide">PURSE TRACKER</h2>
        <span className="text-xs text-ink/40">{sorted.length} Active Teams</span>
      </div>
      <div className="space-y-2">
        {sorted.map((team) => {
          const meta = TEAMS_BY_ID[team.teamId]
          const isMine = team.teamId === myTeamId
          const nickname = nicknameByPlayerId[team.ownerId]
          const isLeader = team.purseLeft === topPurse

          return (
            <div
              key={team.teamId}
              className={`flex items-center justify-between rounded-lg px-3 py-2 bg-mist border
                ${isMine ? 'border-ink' : 'border-line'}`}
            >
              <div className="flex items-center gap-2 min-w-0">
                {isLeader && <span className="text-sm shrink-0">🥇</span>}
                <span
                  className="text-xs font-display px-2 py-0.5 rounded text-white shrink-0"
                  style={{ backgroundColor: meta?.color }}
                >
                  {team.teamId}
                </span>
                <div className="min-w-0">
                  {nickname && (
                    <p className="text-xs text-ink/70 truncate leading-tight">{nickname}</p>
                  )}
                  <p className="text-[11px] text-ink/40 leading-tight">
                    {team.playerCount} players · {team.overseasCount} 🌍
                  </p>
                </div>
              </div>
              <span className="font-display text-ink shrink-0">₹{team.purseLeft}L</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default PurseTracker