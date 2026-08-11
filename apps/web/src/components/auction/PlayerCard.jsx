import { AnimatePresence, motion } from 'framer-motion'
import { useAuctionStore } from '../../store/auctionStore'

const STATS = [
  { label: 'BAT', key: 'bat' },
  { label: 'PWR', key: 'pwr' },
  { label: 'BWL', key: 'bwl' },
  { label: 'TEC', key: 'tec' },
  { label: 'CLT', key: 'clt' }
]

const StatBar = ({ label, value }) => (
  <div className="flex items-center gap-2">
    <span className="text-xs text-paper/50 w-8">{label}</span>
    <div className="flex-1 h-1.5 bg-paper/10 rounded-full overflow-hidden">
      <div className="h-full bg-brand-red rounded-full" style={{ width: `${value}%` }} />
    </div>
    <span className="text-xs text-paper/70 w-6 text-right">{value}</span>
  </div>
)

const PlayerCard = () => {
  const player = useAuctionStore((s) => s.currentPlayer)

  return (
    <AnimatePresence mode="wait">
      {!player ? (
        <motion.div
          key="waiting"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="bg-paper border border-line rounded-2xl p-8 text-center"
        >
          <p className="text-ink/40 font-display text-2xl">Waiting for next player…</p>
        </motion.div>
      ) : (
        <motion.div
          // Keyed by slNo — a new player loading in gets a fresh mount,
          // so the old card exits and the new one enters distinctly
          // rather than just re-rendering its text in place.
          key={player.slNo}
          initial={{ opacity: 0, x: 24 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -24 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
          className="bg-charcoal text-paper rounded-2xl p-6 shadow-xl"
        >
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <p className="text-xs uppercase tracking-widest text-paper/50">{player.role}</p>
              <h1 className="font-display text-4xl tracking-wide leading-tight truncate">
                {player.playerName}
              </h1>
              <p className="text-sm text-paper/60 mt-1">
                {player.country}
                {player.nationality === 'Overseas' && ' · Overseas'}
              </p>
            </div>
            {player.rating != null && (
              <div className="text-right shrink-0">
                <p className="text-xs text-paper/50">RATING</p>
                <p className="font-display text-3xl text-brand-red">{player.rating}</p>
              </div>
            )}
          </div>

          {player.stats && Object.keys(player.stats).length > 0 && (
            <div className="mt-5 grid grid-cols-1 gap-2">
              {STATS.map(({ label, key }) => (
                <StatBar key={key} label={label} value={player.stats[key] ?? 0} />
              ))}
            </div>
          )}

          <div className="mt-5 pt-4 border-t border-paper/10">
            <p className="text-xs text-paper/50">BASE PRICE</p>
            <p className="font-display text-2xl text-paper">₹{player.basePrice}L</p>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default PlayerCard