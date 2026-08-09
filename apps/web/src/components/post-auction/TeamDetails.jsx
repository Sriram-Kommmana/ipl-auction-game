const ROLE_ORDER = ['BATSMAN', 'WICKET KEEPER', 'ALL ROUNDER', 'BOWLER']
const ROLE_LABELS = {
  BATSMAN: 'Batters',
  'WICKET KEEPER': 'Wicket Keepers',
  'ALL ROUNDER': 'All Rounders',
  BOWLER: 'Bowlers'
}

const TeamDetails = ({ team }) => {
  if (!team) return null

  const grouped = ROLE_ORDER
    .map((role) => ({
      role,
      players: team.squad.filter((p) => p.role?.toUpperCase() === role)
    }))
    .filter((g) => g.players.length > 0)

  return (
    <div className="bg-paper border border-line rounded-2xl p-6">
      <h2 className="font-display text-2xl text-ink mb-4 tracking-wide">SQUAD DETAILS</h2>

      {team.squad.length === 0 ? (
        <p className="text-xs text-ink/30 text-center py-4">No players in this squad</p>
      ) : (
        <div className="space-y-4">
          {grouped.map(({ role, players }) => (
            <div key={role}>
              <p className="text-xs uppercase tracking-wide text-ink/40 mb-2">
                {ROLE_LABELS[role]} ({players.length})
              </p>
              <div className="space-y-1.5">
                {players.map((p) => (
                  <div
                    key={p.slNo}
                    className="flex items-center justify-between text-sm bg-mist rounded-lg px-3 py-2"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-ink truncate">{p.playerName}</span>
                      {p.nationality === 'Overseas' && <span className="text-xs">🌍</span>}
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                      <span className="text-xs text-ink/40">⭐ {p.rating}</span>
                      <span className="font-display text-ink">₹{p.boughtFor}L</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default TeamDetails