// apps/web/src/components/auction/PassButton.jsx
import { useState } from 'react'
import { useAuctionStore } from '../../store/auctionStore'
import { useSessionStore } from '../../store/sessionStore'
import socket from '../../lib/socket'

// Solo only — sits in the header controls where multiplayer has Skip.
// Tells the server you're done with this lot: once none of the AI franchises
// will bid again, it closes straight away instead of waiting out the
// countdown. Bidding again after passing is fine — it cancels the pass.
const PassButton = ({ className = '' }) => {
  const playerId = useSessionStore((s) => s.playerId)
  const myTeamId = useSessionStore((s) => s.teamId)
  const currentPlayerIndex = useAuctionStore((s) => s.currentPlayerIndex)
  const isReauction = useAuctionStore((s) => s.isReauction)
  const currentBidderId = useAuctionStore((s) => s.currentBidderId)
  const timerState = useAuctionStore((s) => s.timerState)
  const bidFeed = useAuctionStore((s) => s.bidFeed)

  // Remember WHICH lot you passed on and how far its bid feed had got.
  // "Passed" is derived from that, so a new lot or a later bid of your own
  // (which the server also treats as cancelling the pass) resets it without
  // any effect juggling.
  const [passMark, setPassMark] = useState(null)
  const lotKey = `${isReauction ? 'R' : 'M'}:${currentPlayerIndex}`
  const passed = passMark?.lotKey === lotKey &&
    !bidFeed.slice(passMark.feedLength).some((entry) => entry.teamId === myTeamId)

  const leading = Boolean(currentBidderId) && currentBidderId === myTeamId
  const open = timerState === 'RUNNING'

  const handlePass = () => {
    setPassMark({ lotKey, feedLength: bidFeed.length })
    socket.emit('passLot', { playerId })
  }

  let note = 'Done with this player? Pass and the lot closes as soon as the AI stops bidding.'
  if (leading) note = "You're leading — the lot closes as soon as the AI stops bidding."
  else if (passed) note = 'Passed — closing once the AI franchises stop.'

  return (
    <button
      type="button"
      onClick={handlePass}
      disabled={!open || leading || passed}
      title={note}
      aria-label={`${passed ? 'Passed' : 'Pass'}. ${note}`}
      className={className}
    >
      {passed ? 'PASSED' : <>PASS<span className="hidden sm:inline"> ✕</span></>}
    </button>
  )
}

export default PassButton
