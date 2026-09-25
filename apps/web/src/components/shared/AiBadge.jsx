import { RL_PERSONAS, RULE_PERSONAS } from '@ipl-auction/shared'

// Marks a franchise run by a server-side bot. "AI" = hand-written rule bot,
// "RL" = reinforcement-learning bot. The tooltip names its personality.
const describe = (kind, persona) => {
  const info = kind === 'rl' ? RL_PERSONAS[persona] : RULE_PERSONAS[persona]
  const style = kind === 'rl' ? 'Reinforcement-learning bot' : 'Rule-based bot'
  return info ? `${style} · ${info.name} — ${info.blurb}` : style
}

const AiBadge = ({ kind, persona, className = '' }) => (
  <span
    title={describe(kind, persona)}
    className={`inline-flex items-center font-mono text-[9px] font-bold leading-none tracking-[0.12em]
      px-1 py-[3px] border shrink-0 align-middle
      ${kind === 'rl' ? 'border-red/70 text-red' : 'border-bone/35 text-bone/70'} ${className}`}
  >
    {kind === 'rl' ? 'RL' : 'AI'}
  </span>
)

export default AiBadge
