// apps/web/src/components/post-auction/AuctionHistory.jsx
import { useMemo, useState } from 'react'
import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'

const STATUS_STYLES = {
  sold: { label: 'SOLD', className: 'bg-red text-bone border border-red' },
  unsold: { label: 'UNSOLD', className: 'text-bone/50 border border-line-strong' },
  skipped: { label: 'SKIPPED', className: 'text-amber/80 border border-amber/40 border-dashed' }
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
    <div className="panel p-4 sm:p-6">
      <div className="section-head">
        <span className="section-num">07</span>
        <h2 className="section-title">Auction History</h2>
        <span className="section-jp">入札履歴</span>
      </div>

      <div className="flex items-center justify-between mb-3 flex-wrap gap-3">
        <div className="flex">
          {FILTERS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setStatusFilter(s)}
              data-active={statusFilter === s}
              className="tab text-xs px-3 py-1.5 border border-line -ml-px first:ml-0"
            >
              {s}
            </button>
          ))}
        </div>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search player…"
          className="field !w-auto !py-1.5 !text-sm"
        />
      </div>

      {filtered.length === 0 ? (
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-bone/30 text-center py-4">No matching players</p>
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
                className="row flex flex-wrap sm:flex-nowrap items-center justify-between gap-x-2 gap-y-1.5 text-sm px-3 py-2"
              >
                <div className="flex items-center gap-2 min-w-0 basis-full sm:basis-auto">
                  <span className="font-mono text-[10px] text-bone/30 w-8 shrink-0">#{String(h.auctionOrder).padStart(3, '0')}</span>
                  <span className="text-bone truncate">{h.playerName}</span>
                </div>
                <div className="flex items-center gap-2 shrink-0 ml-10 sm:ml-0">
                  {team && (
                    <span className="team-chip" style={teamChipStyle(team)}>
                      {team.teamId}
                    </span>
                  )}
                  {h.soldFor != null && (
                    <span className="num text-lg text-bone">₹{h.soldFor}L</span>
                  )}
                  <span className={`font-mono text-[9px] tracking-wider px-1.5 py-0.5 min-w-14 text-center ${statusStyle.className}`}>
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