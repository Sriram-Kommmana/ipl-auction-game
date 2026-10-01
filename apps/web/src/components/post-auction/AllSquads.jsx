import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'
import { rankLabel } from '../../lib/results'

const ROLE_ORDER = ['BATSMAN', 'WICKET KEEPER', 'ALL ROUNDER', 'BOWLER']
const ROLE_SHORT = { BATSMAN: 'BAT', 'WICKET KEEPER': 'WK', 'ALL ROUNDER': 'AR', BOWLER: 'BWL' }

const byRoleThenRating = (a, b) =>
  (ROLE_ORDER.indexOf(a.role) - ROLE_ORDER.indexOf(b.role)) || (b.rating - a.rating)

// Every franchise's full squad, in final rank order. Players in the Best XI
// are marked XI, the Impact Player is marked IMP.
const SquadCard = ({ team, isMine }) => {
  const r = team.ranking
  const xi = new Set(r.xi)
  const squad = [...team.squad].sort(byRoleThenRating)

  return (
    <div className={`row !bg-raised/60 p-3 sm:p-4 min-w-0 ${isMine ? '!border-cyan/60 !border-l-4 !border-l-cyan' : '!border-line'}`}>
      <div className="flex items-start justify-between gap-2 pb-2 mb-2 border-b border-line">
        <div className="flex items-center gap-2 min-w-0">
          <span className="num text-xl text-bone/40 shrink-0 leading-none">{rankLabel(r)}</span>
          <span className="team-chip shrink-0" style={teamChipStyle(TEAMS_BY_ID[team.teamId])}>{team.teamId}</span>
          <div className="min-w-0">
            <p className="font-display text-base uppercase tracking-wide text-bone truncate leading-tight flex items-center gap-2">
              <span className="truncate">{team.teamName}</span>
              {isMine && <span className="font-mono text-[9px] tracking-[0.2em] text-cyan border border-cyan/60 px-1 py-0.5 shrink-0">YOU</span>}
            </p>
            <p className="font-mono text-[10px] text-bone/45 truncate">{team.ownerNickname}</p>
          </div>
        </div>
        <div className="text-right shrink-0">
          <p className="num text-2xl text-bone leading-none">{r.score}</p>
          <p className="font-mono text-[9px] text-bone/40 mt-1">SQUAD SCORE</p>
        </div>
      </div>

      <p className="font-mono text-[10px] text-bone/50 mb-2 flex flex-wrap gap-x-3">
        <span>{team.playerCount} PLR · {team.overseasCount} OS</span>
        <span>SPENT <span className="text-bone/80">₹{team.purseSpent}L</span></span>
        <span>LEFT <span className="text-bone/80">₹{team.purseLeft}L</span></span>
      </p>

      {squad.length === 0 ? (
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-bone/30 text-center py-3">No players bought</p>
      ) : (
        <ul className="space-y-0.5">
          {squad.map((p) => {
            const tag = xi.has(p.slNo) ? 'XI' : p.slNo === r.impactPlayer ? 'IMP' : null
            return (
              <li key={p.slNo} className="flex items-center justify-between gap-2 text-xs py-0.5">
                <span className="flex items-center gap-1.5 min-w-0">
                  <span className="font-mono text-[9px] text-bone/35 w-7 shrink-0">{ROLE_SHORT[p.role] ?? p.role}</span>
                  <span className={`truncate ${tag ? 'text-bone' : 'text-bone/55'}`}>{p.playerName}</span>
                  {p.nationality === 'Overseas' && (
                    <span className="font-mono text-[8px] border border-cyan/50 text-cyan px-0.5 shrink-0">OS</span>
                  )}
                  {tag && (
                    <span className={`font-mono text-[8px] px-0.5 shrink-0 ${tag === 'XI' ? 'bg-bone text-void' : 'border border-amber/60 text-amber'}`}>{tag}</span>
                  )}
                </span>
                <span className="flex items-center gap-2 shrink-0">
                  <span className="font-mono text-[10px] text-amber/80">★ {p.rating}</span>
                  <span className="num text-sm text-bone/80 w-14 text-right">₹{p.boughtFor}L</span>
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

const AllSquads = ({ teams, myTeamId, num = '07' }) => (
  <div className="panel p-4 sm:p-6">
    <div className="section-head">
      <span className="section-num">{num}</span>
      <h2 className="section-title">All Squads</h2>
    </div>
    <p className="font-mono text-[10px] leading-relaxed text-bone/40 -mt-2 mb-3">
      Every franchise in final rank order. <span className="bg-bone text-void px-0.5">XI</span> = in the Best XI,{' '}
      <span className="border border-amber/60 text-amber px-0.5">IMP</span> = Impact Player.
    </p>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      {teams.map((team) => (
        <SquadCard key={team.teamId} team={team} isMine={team.teamId === myTeamId} />
      ))}
    </div>
  </div>
)

export default AllSquads
