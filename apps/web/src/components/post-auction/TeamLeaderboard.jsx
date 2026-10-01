import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'
import { TIEBREAK_LABELS, rankLabel } from '../../lib/results'

// Podium tints — gold / bone / red for the top three, muted after that
const RANK_STYLES = ['text-amber', 'text-bone', 'text-red']

// teams: already ranked (rankSquads in packages/shared/src/squadRanking.js)
const TeamLeaderboard = ({ teams, myTeamId, num = '02' }) => (
  <div className="panel p-4 sm:p-6">
    <div className="section-head">
      <span className="section-num">{num}</span>
      <h2 className="section-title">Final Standings</h2>
    </div>
    <p className="font-mono text-[10px] leading-relaxed text-bone/40 -mt-2 mb-3">
      Ranked by Squad Score = 60% matchday strength (best legal XI + Impact Player)
      + 20% injury cover (XI strength with any one starter out) + 20% balance
      (batting + bowling units). Ties are split by XI strength, then injury cover,
      then less purse spent.
    </p>
    <div className="space-y-2">
      {teams.map((team) => {
        const r = team.ranking
        const meta = TEAMS_BY_ID[team.teamId]
        const isMine = team.teamId === myTeamId
        const isWinner = r.rank === 1
        const tieBreak = TIEBREAK_LABELS[r.decidedBy]

        return (
          <div
            key={team.teamId}
            className={`row px-3 sm:px-4 py-3
              ${isWinner ? '!border-amber/50 !border-l-4 !border-l-amber' : ''}
              ${isMine && !isWinner ? '!border-cyan/60 !border-l-4 !border-l-cyan' : ''}
              ${isMine ? 'bg-cyan/[0.06]' : ''}`}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                <span className={`num text-2xl sm:text-3xl w-10 sm:w-12 shrink-0 leading-none ${RANK_STYLES[r.rank - 1] ?? 'text-bone/25'}`}>
                  {rankLabel(r)}
                </span>
                <span className="team-chip shrink-0" style={teamChipStyle(meta)}>
                  {team.teamId}
                </span>
                <div className="min-w-0">
                  <p className="font-display text-base sm:text-lg uppercase tracking-wide text-bone truncate leading-tight flex items-center gap-2">
                    <span className="truncate">{team.teamName}</span>
                    {isMine && <span className="font-mono text-[9px] tracking-[0.2em] text-cyan border border-cyan/60 px-1 py-0.5 shrink-0">YOU</span>}
                  </p>
                  <p className={`font-mono text-[11px] truncate ${isMine ? 'text-cyan/80' : 'text-bone/45'}`}>
                    {team.ownerNickname}
                    <span> · {team.playerCount} PLR · {team.overseasCount} OS</span>
                  </p>
                </div>
              </div>
              <div className="text-right shrink-0">
                <p className={`num text-3xl leading-none ${isWinner ? 'text-amber' : 'text-bone'}`} title="Squad Score">{r.score}</p>
                <p className="font-mono text-[10px] text-bone/40 mt-1">SQUAD SCORE</p>
              </div>
            </div>

            <div className="mt-2 pt-2 border-t border-line flex flex-wrap items-center justify-between gap-x-4 gap-y-1 font-mono text-[10px] text-bone/50">
              <span className="flex flex-wrap gap-x-3 gap-y-1">
                <span>XI <span className="text-bone/80">{r.xiStrength}</span></span>
                <span>MATCHDAY <span className="text-bone/80">{r.matchday}</span></span>
                <span>INJURY COVER <span className="text-bone/80">{r.injuryCover}</span></span>
                <span>BALANCE <span className="text-bone/80">{r.balance}</span></span>
              </span>
              <span className="flex gap-3">
                <span>SPENT <span className="text-bone/80">₹{team.purseSpent}L</span></span>
                <span>LEFT <span className="text-bone/80">₹{team.purseLeft}L</span></span>
              </span>
            </div>

            {(tieBreak || r.tied) && (
              <p className="mt-1.5 font-mono text-[10px] text-amber/80">
                {r.tied && !tieBreak
                  ? 'Level with another team on every tie-breaker — rank shared'
                  : `Level on Squad Score with the team above — placed below it on ${tieBreak}`}
              </p>
            )}
          </div>
        )
      })}
    </div>
  </div>
)

export default TeamLeaderboard
