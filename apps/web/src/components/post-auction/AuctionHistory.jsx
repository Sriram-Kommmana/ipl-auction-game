// apps/web/src/components/post-auction/AuctionHistory.jsx
import { useMemo, useState } from 'react'
import { TEAMS_BY_ID } from '../../constants/teams'

const STATUS_STYLES = {
  sold: { label: 'SOLD', className: 'bg-brand-red text-paper' },
  unsold: { label: 'UNSOLD', className: 'bg-ink/10 text-ink/50' },
  skipped: { label: 'SKIPPED', className: 'bg-mist text-ink/40 border border-line' }
}

const FILTERS = ['all', 'sold', 'unsold', 'skipped']

const AuctionHistory = ({ history }) => {
  const [query, setQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return [...history]
      .sort((a, b) => a.auctionOrder - b.auctionOrder)
      .filter((h) => statusFilter === 'all' || h.status === statusFilter)
      .filter((h) => h.playerName.toLowerCase().includes(q))
  }, [history, query, statusFilter])

  return (
    <div className="bg-paper border border-line rounded-2xl p-6">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h2 className="font-display text-2xl text-ink tracking-wide">AUCTION HISTORY</h2>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search player…"
          className="bg-mist border border-line rounded-lg px-3 py-1.5 text-sm text-ink
                     placeholder:text-ink/30 focus:outline-none focus:ring-2 focus:ring-brand-red"
        />
      </div>

      <div className="flex gap-2 mb-3">
        {FILTERS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setStatusFilter(s)}
            className={`text-xs font-display px-3 py-1 rounded-full transition-colors ${
              statusFilter === s ? 'bg-ink text-paper' : 'bg-mist text-ink/50 hover:text-ink'
            }`}
          >
            {s.toUpperCase()}
          </button>
        ))}
      </div>

      {filtered.length === 0 ? (
        <p className="text-xs text-ink/30 text-center py-4">No matching players</p>
      ) : (
        <div className="space-y-1.5 max-h-96 overflow-y-auto">
          {filtered.map((h) => {
            const team = h.soldTo ? TEAMS_BY_ID[h.soldTo] : null
            const statusStyle = STATUS_STYLES[h.status] || STATUS_STYLES.unsold

            return (
              // auctionOrder, NOT slNo — a skipped-then-resold player
              // appears twice in history with the same slNo.
              <div
                key={h.auctionOrder}
                className="flex items-center justify-between text-sm bg-mist rounded-lg px-3 py-2"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-xs text-ink/30 w-6 shrink-0">#{h.auctionOrder}</span>
                  <span className="text-ink truncate">{h.playerName}</span>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {team && (
                    <span
                      className="text-xs font-display px-2 py-0.5 rounded text-white"
                      style={{ backgroundColor: team.color }}
                    >
                      {team.teamId}
                    </span>
                  )}
                  {h.soldFor != null && (
                    <span className="font-display text-ink">₹{h.soldFor}L</span>
                  )}
                  <span className={`text-[10px] font-display px-1.5 py-0.5 rounded ${statusStyle.className}`}>
                    {statusStyle.label}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default AuctionHistory