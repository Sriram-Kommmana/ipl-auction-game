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
    <div className="panel p-6">
      <div className="section-head">
        <span className="section-num">05</span>
        <h2 className="section-title">Squad Details</h2>
        <span className="section-jp">選手一覧</span>
      </div>

      {team.squad.length === 0 ? (
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-bone/30 text-center py-4">No players in this squad</p>
      ) : (
        <div className="space-y-4">
          {grouped.map(({ role, players }) => (
            <div key={role}>
              <p className="label-mono mb-2 flex items-center gap-2">
                <span className="h-px w-3 bg-red" />
                {ROLE_LABELS[role]} <span className="text-red">[{players.length}]</span>
              </p>
              <div className="space-y-1.5">
                {players.map((p) => (
                  <div
                    key={p.slNo}
                    className="row flex items-center justify-between text-sm px-3 py-2"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-bone truncate">{p.playerName}</span>
                      {p.nationality === 'Overseas' && (
                        <span className="font-mono text-[9px] border border-cyan/50 text-cyan px-1 shrink-0">OS</span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                      <span className="font-mono text-[11px] text-amber/80">★ {p.rating}</span>
                      <span className="num text-lg text-bone">₹{p.boughtFor}L</span>
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