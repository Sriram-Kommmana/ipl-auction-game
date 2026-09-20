import { AnimatePresence, motion } from 'framer-motion'
import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'
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
    <div className="panel p-3 text-center">
      <div className="flex items-center justify-center gap-2">
        <span className="h-1.5 w-1.5 bg-red animate-blink" />
        <p className="label-mono">Current Bid // 現在価格</p>
      </div>

      {/* Keyed by currentBid — every new bid amount re-mounts this element,
          giving a distinct pulse rather than the number just silently
          swapping in place. */}
      <AnimatePresence mode="popLayout">
        <motion.p
          key={currentBid}
          initial={{ scale: 1.25, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
          className="num text-5xl min-[400px]:text-6xl text-bone mt-1 leading-none"
        >
          <span className="text-bone/50">₹</span>{currentBid}<span className="text-red">L</span>
        </motion.p>
      </AnimatePresence>

      {team ? (
        <>
          <div className="flex items-center justify-center gap-2 mt-3 pt-3 border-t border-line">
            <p className="label-mono">Highest Bidder</p>
            <div className="inline-flex items-stretch">
              <span className="team-chip" style={teamChipStyle(team)}>{team.teamId}</span>
              {ownerNickname && (
                <span className="font-mono text-[11px] text-bone bg-raised border border-line px-2 flex items-center">
                  {ownerNickname}
                </span>
              )}
            </div>
          </div>
        </>
      ) : (
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-bone/30 mt-3 pt-3 border-t border-line">— No bids yet —</p>
      )}
    </div>
  )
}

export default CurrentBid