import { TEAMS_BY_ID } from '../../constants/teams'

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
    <div className="bg-paper border border-line rounded-2xl p-6">
      <h2 className="font-display text-2xl text-ink mb-4 tracking-wide">AUCTION SUMMARY</h2>

      <div className="grid grid-cols-3 sm:grid-cols-5 gap-4 mb-4">
        {stats.map((s) => (
          <div key={s.label} className="text-center">
            <p className="font-display text-3xl text-ink">{s.value}</p>
            <p className="text-xs text-ink/50 uppercase tracking-wide mt-1">{s.label}</p>
          </div>
        ))}
      </div>

      {mostExpensive && (
        <div className="border-t border-line pt-4 flex items-center justify-between flex-wrap gap-2">
          <div>
            <p className="text-xs uppercase tracking-wide text-ink/50">Most Expensive Buy</p>
            <p className="font-display text-xl text-ink">{mostExpensive.playerName}</p>
          </div>
          <div className="flex items-center gap-2">
            {mostExpensiveTeam && (
              <span
                className="text-xs font-display px-2 py-0.5 rounded text-white"
                style={{ backgroundColor: TEAMS_BY_ID[mostExpensiveTeam.teamId]?.color }}
              >
                {mostExpensiveTeam.teamId}
              </span>
            )}
            <span className="font-display text-2xl text-brand-red">
              ₹{mostExpensive.soldFor}L
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export default AuctionSummary