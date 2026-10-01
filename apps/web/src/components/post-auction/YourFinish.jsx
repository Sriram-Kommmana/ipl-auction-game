import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'
import { ordinal } from '../../lib/results'

// "You finished 4th of 10" — only when this browser owned a team in the room.
const YourFinish = ({ team, teamCount }) => {
  if (!team) return null
  const { rank, tied, score, xiStrength } = team.ranking

  return (
    <div className="panel border-l-4 !border-l-cyan p-4 sm:p-5 flex flex-wrap items-center justify-between gap-4">
      <div className="flex items-center gap-3 min-w-0">
        <span className="team-chip !text-xs !px-3 !py-2 shrink-0" style={teamChipStyle(TEAMS_BY_ID[team.teamId])}>
          {team.teamId}
        </span>
        <div className="min-w-0">
          <p className="label-mono text-cyan">Your franchise</p>
          <p className="font-display text-3xl sm:text-4xl uppercase leading-none text-bone mt-1">
            You finished <span className={rank === 1 ? 'text-amber' : 'text-cyan'}>{tied ? 'joint ' : ''}{ordinal(rank)}</span> of {teamCount}
          </p>
        </div>
      </div>
      <div className="flex gap-5 shrink-0">
        <div className="text-right">
          <p className="label-mono">Squad Score</p>
          <p className="num text-3xl text-bone leading-none mt-1">{score}</p>
        </div>
        <div className="text-right">
          <p className="label-mono">XI strength</p>
          <p className="num text-3xl text-bone/70 leading-none mt-1">{xiStrength}</p>
        </div>
        <div className="text-right">
          <p className="label-mono">Purse left</p>
          <p className="num text-3xl text-bone/70 leading-none mt-1">₹{team.purseLeft}L</p>
        </div>
      </div>
    </div>
  )
}

export default YourFinish
