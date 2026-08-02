import { Crown } from 'lucide-react'
import { TEAMS_BY_ID } from '../../constants/teams'
import { useRoomStore } from '../../store/roomStore'

const PlayerList = () => {
  const players = useRoomStore((s) => s.players)

  return (
    <div>
      <h2 className="font-display text-2xl text-ink mb-3 tracking-wide">
        PLAYERS ({players.length})
      </h2>
      <div className="space-y-2">
        {players.map((p) => {
          const team = p.teamId ? TEAMS_BY_ID[p.teamId] : null
          return (
            <div
              key={p.playerId}
              className="flex items-center justify-between bg-paper border border-line rounded-lg px-3 py-2"
            >
              <div className="flex items-center gap-2 min-w-0">
                <span
                  className={`w-2 h-2 rounded-full shrink-0 ${
                    p.status === 'online' ? 'bg-green-500' : 'bg-ink/20'
                  }`}
                />
                <span className="text-sm text-ink truncate">{p.nickname}</span>
                {p.isManager && <Crown size={14} className="text-brand-red shrink-0" />}
              </div>
              {team ? (
                <span
                  className="text-xs font-display px-2 py-0.5 rounded text-white shrink-0"
                  style={{ backgroundColor: team.color }}
                >
                  {team.teamId}
                </span>
              ) : (
                <span className="text-xs text-ink/30 shrink-0">No team</span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default PlayerList