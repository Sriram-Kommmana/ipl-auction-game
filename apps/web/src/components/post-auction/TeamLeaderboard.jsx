import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'
import AiBadge from '../shared/AiBadge'

// Podium tints — gold / bone / red for the top three, muted after that
const RANK_STYLES = ['text-amber', 'text-bone', 'text-red']

const TeamLeaderboard = ({ teams }) => {
  const sorted = [...teams].sort((a, b) => b.teamRating - a.teamRating)

  return (
    <div className="panel p-4 sm:p-6">
      <div className="section-head">
        <span className="section-num">02</span>
        <h2 className="section-title">Leaderboard</h2>
        <span className="section-jp">順位</span>
      </div>
      <p className="font-mono text-[10px] leading-relaxed text-bone/40 -mt-2 mb-3">
        Ranked by XI strength — the average rating of the best legal playing XI
        (max 4 overseas, a keeper, 5 bowling options). Empty slots count as 0.
      </p>
      <div className="space-y-2">
        {sorted.map((team, i) => {
          const meta = TEAMS_BY_ID[team.teamId]

          return (
            <div
              key={team.teamId}
              className={`row flex items-center justify-between gap-2 px-3 sm:px-4 py-3
                ${i === 0 ? '!border-amber/50 !border-l-4 !border-l-amber' : ''}`}
            >
              <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                <span className={`num text-2xl sm:text-3xl w-8 sm:w-10 shrink-0 leading-none ${RANK_STYLES[i] ?? 'text-bone/25'}`}>
                  {String(i + 1).padStart(2, '0')}
                </span>
                <span className="team-chip shrink-0" style={teamChipStyle(meta)}>
                  {team.teamId}
                </span>
                <div className="min-w-0">
                  <p className="font-display text-base sm:text-lg uppercase tracking-wide text-bone truncate leading-tight">{team.teamName}</p>
                  <p className="font-mono text-[11px] text-bone/45 truncate flex items-center gap-1.5">
                    {team.isBot && <AiBadge kind={team.botKind} persona={team.botPersona} />}
                    {team.ownerNickname}
                    <span className="sm:hidden"> · {team.playerCount} PLR · {team.overseasCount} OS</span>
                  </p>
                </div>
              </div>
              <div className="text-right shrink-0">
                <p className={`num text-3xl leading-none ${i === 0 ? 'text-amber' : 'text-bone'}`} title="XI strength">{team.teamRating}</p>
                <p className="hidden sm:block font-mono text-[10px] text-bone/40 mt-1">
                  {team.playerCount} PLR · {team.overseasCount} OS
                </p>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default TeamLeaderboard