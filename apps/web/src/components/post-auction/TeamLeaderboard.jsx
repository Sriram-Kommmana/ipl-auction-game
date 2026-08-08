import { TEAMS_BY_ID } from '../../constants/teams'

const MEDALS = ['🥇', '🥈', '🥉']

const TeamLeaderboard = ({ teams }) => {
  const sorted = [...teams].sort((a, b) => b.teamRating - a.teamRating)

  return (
    <div className="bg-paper border border-line rounded-2xl p-6">
      <h2 className="font-display text-2xl text-ink mb-4 tracking-wide">LEADERBOARD</h2>
      <div className="space-y-2">
        {sorted.map((team, i) => {
          const meta = TEAMS_BY_ID[team.teamId]

          return (
            <div
              key={team.teamId}
              className="flex items-center justify-between bg-mist border border-line rounded-xl px-4 py-3"
            >
              <div className="flex items-center gap-3 min-w-0">
                <span className="font-display text-lg text-ink/40 w-6 text-center shrink-0">
                  {MEDALS[i] ?? i + 1}
                </span>
                <span
                  className="text-xs font-display px-2 py-1 rounded text-white shrink-0"
                  style={{ backgroundColor: meta?.color }}
                >
                  {team.teamId}
                </span>
                <div className="min-w-0">
                  <p className="text-ink truncate">{team.teamName}</p>
                  <p className="text-xs text-ink/50 truncate">{team.ownerNickname}</p>
                </div>
              </div>
              <div className="text-right shrink-0">
                <p className="font-display text-xl text-ink">{team.teamRating}</p>
                <p className="text-xs text-ink/40">
                  {team.playerCount} players · {team.overseasCount} 🌍
                </p>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default TeamLeaderboard