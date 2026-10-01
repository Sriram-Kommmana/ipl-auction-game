const formatDuration = (startedAt, completedAt) => {
  const ms = new Date(completedAt) - new Date(startedAt)
  const totalMinutes = Math.round(ms / 60000)
  const hours = Math.floor(totalMinutes / 60)
  const minutes = totalMinutes % 60
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`
}

const AuctionSummary = ({ results, num = '01' }) => {
  const { history, teams, startedAt, completedAt } = results

  const soldEntries = history.filter((h) => h.status === 'sold')
  const unsoldCount = history.filter((h) => h.status === 'unsold').length
  const skippedCount = history.filter((h) => h.status === 'skipped').length

  const stats = [
    { label: 'Duration', value: formatDuration(startedAt, completedAt) },
    { label: 'Sold', value: soldEntries.length },
    { label: 'Unsold', value: unsoldCount },
    { label: 'Skipped', value: skippedCount },
    { label: 'Teams', value: teams.length }
  ]

  return (
    <div className="panel p-4 sm:p-6">
      <div className="section-head">
        <span className="section-num">{num}</span>
        <h2 className="section-title">Auction Summary</h2>
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-5 border-l border-t border-line">
        {stats.map((s) => (
          <div key={s.label} className="border-r border-b border-line px-3 py-4">
            <p className="label-mono">{s.label}</p>
            <p className="num text-3xl sm:text-4xl text-bone mt-2 leading-none whitespace-nowrap">{s.value}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default AuctionSummary