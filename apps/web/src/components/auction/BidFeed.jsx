import { useMemo } from 'react'
import { TEAMS_BY_ID } from '../../constants/teams'
import { useAuctionStore } from '../../store/auctionStore'
import { useRoomStore } from '../../store/roomStore'

const BidFeed = () => {
  const bidFeed = useAuctionStore((s) => s.bidFeed)
  const teams = useRoomStore((s) => s.teams)
  const players = useRoomStore((s) => s.players)

  const ownerIdByTeamId = useMemo(
    () => Object.fromEntries(teams.map((t) => [t.teamId, t.ownerId])),
    [teams]
  )
  const nicknameByPlayerId = useMemo(
    () => Object.fromEntries(players.map((p) => [p.playerId, p.nickname])),
    [players]
  )
  const feed = useMemo(() => [...bidFeed].reverse(), [bidFeed])

  if (bidFeed.length === 0) {
    return (
      <div className="bg-paper border border-line rounded-2xl p-4">
        <h2 className="font-display text-lg text-ink mb-2 tracking-wide">BID FEED</h2>
        <p className="text-xs text-ink/30 text-center py-4">No bids yet on this player</p>
      </div>
    )
  }

  return (
    <div className="bg-paper border border-line rounded-2xl p-4">
      <h2 className="font-display text-lg text-ink mb-2 tracking-wide">BID FEED</h2>
      <div className="space-y-1.5 max-h-72 overflow-y-auto">
        {feed.map((entry, i) => {
          const team = TEAMS_BY_ID[entry.teamId]
          const ownerId = ownerIdByTeamId[entry.teamId]
          const nickname = ownerId ? nicknameByPlayerId[ownerId] : null

          return (
            <div key={i} className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-1.5 min-w-0">
                <span
                  className="text-xs font-display px-2 py-0.5 rounded text-white shrink-0"
                  style={{ backgroundColor: team?.color }}
                >
                  {entry.teamId}
                </span>
                {nickname && <span className="text-xs text-ink/50 truncate">{nickname}</span>}
              </div>
              <span className="font-display text-ink shrink-0">₹{entry.bid}L</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default BidFeed