import { RL_PERSONAS, RULE_PERSONAS } from '@ipl-auction/shared'

// Solo lobby: who you're about to play against. The nine AI franchises take
// whichever teams you don't — seats are assigned when the auction starts.
const Group = ({ title, personas }) => (
  <div>
    <p className="label-mono mb-2">{title}</p>
    <div className="space-y-1.5">
      {Object.values(personas).map((p) => (
        <div key={p.id} className="row px-3 py-2">
          <span className="font-display text-base uppercase tracking-wide text-bone leading-none">{p.name}</span>
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
    </div>
    <div className="space-y-4">
      <Group title="Rule-based" personas={RULE_PERSONAS} />
      <Group title="Reinforcement learning" personas={RL_PERSONAS} />
    </div>
  </div>
)

export default OpponentsPanel
