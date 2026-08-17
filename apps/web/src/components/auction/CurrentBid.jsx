import { AnimatePresence, motion } from 'framer-motion'
import { TEAMS_BY_ID } from '../../constants/teams'
import { useAuctionStore } from '../../store/auctionStore'
import { useRoomStore } from '../../store/roomStore'

const CurrentBid = () => {
  const currentBid = useAuctionStore((s) => s.currentBid)
  const currentBidderId = useAuctionStore((s) => s.currentBidderId)
  const teams = useRoomStore((s) => s.teams)
  const players = useRoomStore((s) => s.players)

  const team = currentBidderId ? TEAMS_BY_ID[currentBidderId] : null
  const teamData = currentBidderId ? teams.find((t) => t.teamId === currentBidderId) : null
  const ownerNickname = teamData
    ? players.find((p) => p.playerId === teamData.ownerId)?.nickname
    : null

  return (
    <div className="bg-paper border border-line rounded-2xl p-2 text-center">
      <p className="text-xs uppercase tracking-widest text-ink/50">Current Bid</p>

      {/* Keyed by currentBid — every new bid amount re-mounts this element,
          giving a distinct pulse rather than the number just silently
          swapping in place. */}
      <AnimatePresence mode="popLayout">
        <motion.p
          key={currentBid}
          initial={{ scale: 1.25, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
          className="font-display text-5xl text-ink mt-1"
        >
          ₹{currentBid}L
        </motion.p>
      </AnimatePresence>

      {team ? (
        <>
          <div className="flex items-baseline justify-center gap-2 mt-3">
            <p className="text-xs uppercase tracking-widest text-ink/40 mt-4">Highest Bidder</p>
            <div
              className="inline-flex items-center gap-2 mt-1 px-3 py-1 rounded-full"
              style={{ backgroundColor: team.color }}
            >
              <span className="text-xs font-display text-white tracking-wide">{team.teamId}</span>
              {ownerNickname && <span className="text-xs text-white/80">· {ownerNickname}</span>}
            </div>
          </div>
        </>
      ) : (
        <p className="text-xs text-ink/30 mt-3">No bids yet</p>
      )}
    </div>
  )
}

export default CurrentBid