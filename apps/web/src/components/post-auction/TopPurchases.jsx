import { useMemo } from 'react'
import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'

const RANK_STYLES = ['text-amber', 'text-bone', 'text-red']

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
    <div className="panel p-4 sm:p-6">
      <div className="section-head">
        <span className="section-num">03</span>
        <h2 className="section-title">Top Purchases</h2>
        <span className="section-jp">高額契約</span>
      </div>

      {top10.length === 0 ? (
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-bone/30 text-center py-4">No players bought yet</p>
      ) : (
        <div className="space-y-1.5">
          {top10.map((p, i) => {
            const team = TEAMS_BY_ID[p.teamId]

            return (
              <div
                key={p.slNo}
                className="row flex items-center justify-between gap-2 text-sm px-3 py-2"
              >
                <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                  <span className={`num text-xl w-7 shrink-0 ${RANK_STYLES[i] ?? 'text-bone/25'}`}>
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <div className="flex flex-col items-start sm:flex-row sm:items-center gap-1 sm:gap-3 min-w-0">
                    <span className="text-bone truncate max-w-full">{p.playerName}</span>
                    <span className="team-chip shrink-0" style={teamChipStyle(team)}>
                      {p.teamId}
                    </span>
                  </div>
                </div>
                <span className={`num text-xl shrink-0 ${i === 0 ? 'text-red' : 'text-bone'}`}>₹{p.boughtFor}L</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default TopPurchases