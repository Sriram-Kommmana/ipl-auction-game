import { useMemo } from 'react'
import { TEAMS_BY_ID } from '../../constants/teams'

const MEDALS = ['🥇', '🥈', '🥉']

const TopPurchases = ({ teams }) => {
  // Reads from each team's FINAL squad (not history) — a player appears
  // exactly once here regardless of skip/re-auction history, so slNo is
  // a safe, unique key.
  const top10 = useMemo(() => {
    const allPlayers = teams.flatMap((team) =>
      team.squad.map((p) => ({ ...p, teamId: team.teamId }))
    )
    return [...allPlayers].sort((a, b) => (b.boughtFor ?? 0) - (a.boughtFor ?? 0)).slice(0, 10)
  }, [teams])

  return (
    <div className="bg-paper border border-line rounded-2xl p-6">
      <h2 className="font-display text-2xl text-ink mb-4 tracking-wide">TOP PURCHASES</h2>

      {top10.length === 0 ? (
        <p className="text-xs text-ink/30 text-center py-4">No players bought yet</p>
      ) : (
        <div className="space-y-1.5">
          {top10.map((p, i) => {
            const team = TEAMS_BY_ID[p.teamId]

            return (
              <div
                key={p.slNo}
                className="flex items-center justify-between text-sm bg-mist rounded-lg px-3 py-2"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <span className="font-display text-ink/40 w-6 text-center shrink-0">
                    {MEDALS[i] ?? i + 1}
                  </span>
                  <span className="text-ink truncate">{p.playerName}</span>
                  <span
                    className="text-xs font-display px-2 py-0.5 rounded text-white shrink-0"
                    style={{ backgroundColor: team?.color }}
                  >
                    {p.teamId}
                  </span>
                </div>
                <span className="font-display text-ink shrink-0">₹{p.boughtFor}L</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default TopPurchases