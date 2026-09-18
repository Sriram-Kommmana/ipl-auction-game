import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'

const formatDuration = (startedAt, completedAt) => {
  const ms = new Date(completedAt) - new Date(startedAt)
  const totalMinutes = Math.round(ms / 60000)
  const hours = Math.floor(totalMinutes / 60)
  const minutes = totalMinutes % 60
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`
}

const AuctionSummary = ({ results }) => {
  const { history, teams, startedAt, completedAt } = results

  const soldEntries = history.filter((h) => h.status === 'sold')
  const unsoldCount = history.filter((h) => h.status === 'unsold').length
  const skippedCount = history.filter((h) => h.status === 'skipped').length

  const mostExpensive = soldEntries.reduce(
    (max, entry) => (entry.soldFor > (max?.soldFor ?? -1) ? entry : max),
    null
  )
  const mostExpensiveTeam = mostExpensive
    ? teams.find((t) => t.teamId === mostExpensive.soldTo)
    : null

  const stats = [
    { label: 'Duration', value: formatDuration(startedAt, completedAt) },
    { label: 'Sold', value: soldEntries.length },
    { label: 'Unsold', value: unsoldCount },
    { label: 'Skipped', value: skippedCount },
    { label: 'Teams', value: teams.length }
  ]

  return (
    <div className="panel p-6">
      <div className="section-head">
        <span className="section-num">01</span>
        <h2 className="section-title">Auction Summary</h2>
        <span className="section-jp">概要</span>
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-5 border-l border-t border-line mb-5">
        {stats.map((s) => (
          <div key={s.label} className="border-r border-b border-line px-3 py-4">
            <p className="label-mono">{s.label}</p>
            <p className="num text-3xl sm:text-4xl text-bone mt-2 leading-none whitespace-nowrap">{s.value}</p>
          </div>
        ))}
      </div>

      {mostExpensive && (
        <div className="bg-red/10 border border-red/40 border-l-4 border-l-red px-4 py-3 flex items-center justify-between flex-wrap gap-2">
          <div>
            <p className="label-mono text-red">★ Most Expensive Buy // 最高額</p>
            <p className="font-display text-3xl uppercase tracking-wide text-bone mt-1">{mostExpensive.playerName}</p>
          </div>
          <div className="flex items-center gap-2">
            {mostExpensiveTeam && (
              <span className="team-chip" style={teamChipStyle(TEAMS_BY_ID[mostExpensiveTeam.teamId])}>
                {mostExpensiveTeam.teamId}
              </span>
            )}
            <span className="num text-4xl text-red glow-red">
              ₹{mostExpensive.soldFor}L
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export default AuctionSummary