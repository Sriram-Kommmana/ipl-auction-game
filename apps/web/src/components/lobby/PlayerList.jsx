import { Crown } from 'lucide-react'
import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'
import { useRoomStore } from '../../store/roomStore'

const PlayerList = () => {
  const players = useRoomStore((s) => s.players)

  return (
    <div className="panel p-4">
      <div className="section-head">
        <span className="section-num">02</span>
        <h2 className="section-title">Players</h2>
        <span className="font-mono text-xs text-red">[{String(players.length).padStart(2, '0')}]</span>
        <span className="section-jp">選手</span>
      </div>
      <div className="space-y-1.5">
        {players.map((p) => {
          const team = p.teamId ? TEAMS_BY_ID[p.teamId] : null
          const isOnline = p.status === 'online'
          return (
            <div
              key={p.playerId}
              className="row flex items-center justify-between px-3 py-2.5"
            >
              <div className="flex items-center gap-2.5 min-w-0">
                <span
                  className={`w-1.5 h-1.5 shrink-0 ${
                    isOnline ? 'bg-cyan shadow-[0_0_8px_var(--color-cyan)]' : 'bg-bone/20'
                  }`}
                  title={isOnline ? 'Online' : 'Offline'}
                />
                <span className={`text-sm truncate ${isOnline ? 'text-bone' : 'text-bone/40'}`}>
                  {p.nickname}
                </span>
                {p.isManager && <Crown size={13} className="text-amber shrink-0" />}
              </div>
              {team ? (
                <span className="team-chip shrink-0" style={teamChipStyle(team)}>
                  {team.teamId}
                </span>
              ) : (
                <span className="font-mono text-[10px] uppercase tracking-wider text-bone/30 shrink-0">
                  — No team
                </span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default PlayerList
