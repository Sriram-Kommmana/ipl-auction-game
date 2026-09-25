import { useMemo } from 'react'
import { selectBestXI } from '@ipl-auction/shared'

const ROLE_SHORT = {
  BATSMAN: 'BAT',
  BOWLER: 'BWL',
  'ALL ROUNDER': 'AR',
  'WICKET KEEPER': 'WK'
}

// The XI is chosen by the same optimiser the server scores with
// (packages/shared/src/scoring.js): best total rating under the playing-XI
// rules — max 4 overseas, at least 1 keeper and 5 bowling options. Roles
// the squad can't cover show up as empty slots.
const BestXI = ({ team }) => {
  const result = useMemo(() => (team ? selectBestXI(team.squad) : null), [team])
  const xi = result?.players ?? []

  if (!team) return null

  return (
    <div className="panel p-4 sm:p-6">
      <div className="section-head">
        <span className="section-num">06</span>
        <h2 className="section-title">Best XI</h2>
        <span className="section-jp">最強布陣</span>
      </div>

      {xi.length === 0 ? (
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-bone/30 text-center py-4">Not enough players for a Best XI yet</p>
      ) : (
        <div className="space-y-1.5">
          {xi.map((p, i) => (
            <div
              key={p.slNo}
              className="row flex items-center justify-between text-sm px-3 py-2"
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className="font-mono text-[10px] text-bone/30 w-5 shrink-0">{String(i + 1).padStart(2, '0')}</span>
                {/*min-w-[48px] => min-w-12 */}
                <span className="font-mono text-[10px] font-bold bg-bone text-void px-1.5 py-0.5 shrink-0 min-w-12 text-center">
                  {ROLE_SHORT[p.role] ?? p.role}
                </span>
                <span className="text-bone truncate">{p.playerName}</span>
              </div>
              <span className="font-mono text-[11px] text-amber/80 shrink-0">★ {p.rating}</span>
            </div>
          ))}
          {Array.from({ length: result.emptySlots }, (_, i) => (
            <div
              key={`empty-${i}`}
              className="row flex items-center justify-between text-sm px-3 py-2 opacity-50"
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className="font-mono text-[10px] text-bone/30 w-5 shrink-0">{String(xi.length + i + 1).padStart(2, '0')}</span>
                <span className="font-mono text-[10px] uppercase tracking-[0.15em] text-red">Empty slot</span>
              </div>
              <span className="font-mono text-[11px] text-bone/30 shrink-0">★ 0</span>
            </div>
          ))}
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-bone/45 text-right pt-1">
            XI strength <span className="text-amber">{result.strength}</span>
          </p>
        </div>
      )}
    </div>
  )
}

export default BestXI