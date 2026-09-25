import { RL_PERSONAS, RULE_PERSONAS } from '@ipl-auction/shared'
import AiBadge from '../shared/AiBadge'

// Solo lobby: who you're about to play against. The nine AI franchises take
// whichever teams you don't — seats are assigned when the auction starts.
const Group = ({ title, kind, personas }) => (
  <div>
    <p className="label-mono mb-2">{title}</p>
    <div className="space-y-1.5">
      {Object.values(personas).map((p) => (
        <div key={p.id} className="row px-3 py-2">
          <div className="flex items-center gap-2">
            <AiBadge kind={kind} persona={p.id} />
            <span className="font-display text-base uppercase tracking-wide text-bone leading-none">{p.name}</span>
          </div>
          <p className="text-xs text-bone/50 mt-1 leading-snug">{p.blurb}</p>
        </div>
      ))}
    </div>
  </div>
)

const OpponentsPanel = () => (
  <div className="panel p-4 flex flex-col">
    <div className="section-head">
      <span className="section-num">02</span>
      <h2 className="section-title">Opponents</h2>
      <span className="section-jp">対戦相手</span>
    </div>
    <div className="space-y-4">
      <Group title="Rule-based" kind="rule" personas={RULE_PERSONAS} />
      <Group title="Reinforcement learning" kind="rl" personas={RL_PERSONAS} />
    </div>
  </div>
)

export default OpponentsPanel
