import { useMemo } from 'react'

const ROLE_SHORT = {
  BATSMAN: 'BAT',
  BOWLER: 'BWL',
  'ALL ROUNDER': 'AR',
  'WICKET KEEPER': 'WK'
}

// Greedy selection, per spec: 4 Batters (highest BAT), 1 WK (best rating),
// 2 All-Rounders (best BAT+BWL combined), 3 Bowlers (highest BWL),
// 1 extra (best remaining by rating). A Set tracks who's already picked
// so nobody gets double-counted across categories. Gracefully handles
// small squads — slice() on a short array just returns what's available.
const pickBestXI = (squad) => {
  const used = new Set()

  const pick = (pool, count, sortFn) => {
    const sorted = [...pool].filter((p) => !used.has(p.slNo)).sort(sortFn)
    const picked = sorted.slice(0, count)
    picked.forEach((p) => used.add(p.slNo))
    return picked
  }

  const batters = squad.filter((p) => p.role === 'BATSMAN')
  const bowlers = squad.filter((p) => p.role === 'BOWLER')
  const allRounders = squad.filter((p) => p.role === 'ALL ROUNDER')
  const keepers = squad.filter((p) => p.role === 'WICKET KEEPER')

  const bestBatters = pick(batters, 4, (a, b) => (b.stats?.bat ?? 0) - (a.stats?.bat ?? 0))
  const bestKeeper = pick(keepers, 1, (a, b) => (b.rating ?? 0) - (a.rating ?? 0))
  const bestARs = pick(
    allRounders,
    2,
    (a, b) =>
      ((b.stats?.bat ?? 0) + (b.stats?.bwl ?? 0)) - ((a.stats?.bat ?? 0) + (a.stats?.bwl ?? 0))
  )
  const bestBowlers = pick(bowlers, 3, (a, b) => (b.stats?.bwl ?? 0) - (a.stats?.bwl ?? 0))

  const remaining = squad.filter((p) => !used.has(p.slNo))
  const extra = pick(remaining, 1, (a, b) => (b.rating ?? 0) - (a.rating ?? 0))

  return [...bestBatters, ...bestKeeper, ...bestARs, ...bestBowlers, ...extra]
}

const BestXI = ({ team }) => {
  const xi = useMemo(() => (team ? pickBestXI(team.squad) : []), [team])

  if (!team) return null

  return (
    <div className="bg-paper border border-line rounded-2xl p-6">
      <h2 className="font-display text-2xl text-ink mb-4 tracking-wide">BEST XI</h2>

      {xi.length === 0 ? (
        <p className="text-xs text-ink/30 text-center py-4">Not enough players for a Best XI yet</p>
      ) : (
        <div className="space-y-1.5">
          {xi.map((p) => (
            <div
              key={p.slNo}
              className="flex items-center justify-between text-sm bg-mist rounded-lg px-3 py-2"
            >
              <div className="flex items-center gap-2 min-w-0">
                {/*min-w-[48px] => min-w-12 */}
                <span className="text-[10px] font-display bg-ink text-paper px-1.5 py-0.5 rounded shrink-0 min-w-12 text-center">
                  {ROLE_SHORT[p.role] ?? p.role}
                </span>
                <span className="text-ink truncate">{p.playerName}</span>
              </div>
              <span className="text-xs text-ink/40 shrink-0">⭐ {p.rating}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default BestXI