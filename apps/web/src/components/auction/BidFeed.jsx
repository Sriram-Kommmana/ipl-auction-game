import { useMemo } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'
import { useAuctionStore } from '../../store/auctionStore'
import { useRoomStore } from '../../store/roomStore'
import AiBadge from '../shared/AiBadge'

const BidFeed = () => {
  const bidFeed = useAuctionStore((s) => s.bidFeed)
  const teams = useRoomStore((s) => s.teams)
  const players = useRoomStore((s) => s.players)

  const ownerIdByTeamId = useMemo(
    () => Object.fromEntries(teams.map((t) => [t.teamId, t.ownerId])),
    [teams]
  )
  const playerById = useMemo(
    () => Object.fromEntries(players.map((p) => [p.playerId, p])),
    [players]
  )
  const feed = useMemo(() => [...bidFeed].reverse(), [bidFeed])

  if (bidFeed.length === 0) {
    return (
      <div className="panel p-4 h-full">
        <div className="section-head">
          <span className="section-num">02</span>
          <h2 className="section-title">Bid Feed</h2>
          <span className="section-jp">入札記録</span>
        </div>
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-bone/30 text-center py-4">No bids yet on this player</p>
      </div>
    )
  }

  return (
    <div className="panel p-4 h-full flex flex-col">
      <div className="section-head">
        <span className="section-num">02</span>
        <h2 className="section-title">Bid Feed</h2>
        <span className="section-jp">入札記録</span>
      </div>
      <div className="space-y-1.5 flex-1 min-h-0 overflow-y-auto pr-1">
        {/* initial={false} — don't replay entrance animations for items
            that already existed when this list first mounted/reset for a
            new player. Only genuinely NEW bids get the slide-in. */}
        <AnimatePresence initial={false}>
          {feed.map((entry) => {
            const team = TEAMS_BY_ID[entry.teamId]
            const ownerId = ownerIdByTeamId[entry.teamId]
            const owner = ownerId ? playerById[ownerId] : null
            const nickname = owner?.nickname

            return (
              // key={entry.bid} — bid amounts strictly increase within one
              // player's lot and never repeat, so this is a genuinely
              // stable, unique identity per entry (unlike array index,
              // which would make AnimatePresence think every item changed
              // whenever a new bid is prepended).
              <motion.div
                key={entry.bid}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="row flex items-center justify-between text-sm px-2.5 py-1.5 first:!border-l-2 first:!border-l-red"
              >
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="team-chip shrink-0" style={teamChipStyle(team)}>
                    {entry.teamId}
                  </span>
                  {owner?.isBot && <AiBadge kind={owner.botKind} persona={owner.botPersona} />}
                  {nickname && <span className="font-mono text-[11px] text-bone/50 truncate">{nickname}</span>}
                </div>
                <span className="num text-lg text-bone shrink-0">₹{entry.bid}L</span>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>
    </div>
  )
}

export default BidFeed