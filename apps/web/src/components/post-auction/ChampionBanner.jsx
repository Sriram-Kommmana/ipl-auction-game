import { motion, useReducedMotion } from 'framer-motion'
import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'

// The #1 team(s) from the final ranking — the leaderboard stays the
// authoritative standings; this only celebrates the top of it.
const ChampionBanner = ({ champions, myTeamId }) => {
  const reduceMotion = useReducedMotion()
  if (!champions?.length) return null
  const joint = champions.length > 1

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: 'spring', stiffness: 260, damping: 24 }}
      className="panel relative overflow-hidden !border-amber/40 shadow-[6px_6px_0_0_var(--color-amber)]"
      aria-label={joint ? 'Joint auction champions' : 'Auction champion'}
    >
      <div className="h-1 bg-amber" />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgb(255_176_32/0.14),transparent_60%)]"
      />
      <div className="relative p-4 sm:p-6 space-y-4">
        <p className="label-mono text-amber flex items-center gap-2">
          <span aria-hidden className="text-base leading-none">🏆</span>
          {joint ? 'Joint auction champions' : 'Auction champion'}
        </p>

        {champions.map((team) => (
          <div key={team.teamId} className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
            <div className="flex items-center gap-3 min-w-0 flex-1">
              <span className="team-chip !text-sm !px-3 !py-2.5 shrink-0" style={teamChipStyle(TEAMS_BY_ID[team.teamId])}>
                {team.teamId}
              </span>
              <div className="min-w-0">
                <h2 className="font-display text-3xl sm:text-5xl uppercase leading-[0.9] text-bone truncate">
                  {team.teamName}
                </h2>
                <p className="font-mono text-xs text-bone/60 mt-1 truncate flex items-center gap-2">
                  <span className="truncate">{team.ownerNickname}</span>
                  {team.teamId === myTeamId && (
                    <span className="font-mono text-[9px] tracking-[0.2em] text-cyan border border-cyan/60 px-1 py-0.5 shrink-0">YOU</span>
                  )}
                </p>
              </div>
            </div>
            <div className="flex gap-6 shrink-0">
              <div>
                <p className="label-mono">Squad Score</p>
                <p className="num text-4xl text-amber leading-none mt-1">{team.ranking.score}</p>
              </div>
              <div>
                <p className="label-mono">XI Strength</p>
                <p className="num text-4xl text-bone/80 leading-none mt-1">{team.ranking.xiStrength}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </motion.section>
  )
}

export default ChampionBanner
