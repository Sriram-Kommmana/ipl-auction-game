// TEMPORARY placeholder — real Auction UI comes next in the build order.
import { useParams } from 'react-router-dom'
import { useAuctionStore } from '../store/auctionStore'

const Auction = () => {
  const { roomId } = useParams()
  const currentPlayer = useAuctionStore((s) => s.currentPlayer)
  const currentBid = useAuctionStore((s) => s.currentBid)
  const timerState = useAuctionStore((s) => s.timerState)

  return (
    <div className="min-h-screen bg-mist p-8">
      <h1 className="font-display text-4xl text-ink">AUCTION — {roomId}</h1>
      <p className="text-ink/60 mt-1">timerState: {timerState}</p>
      <pre className="text-xs mt-4 bg-paper p-4 rounded-lg border border-line overflow-auto">
        {JSON.stringify({ currentPlayer, currentBid }, null, 2)}
      </pre>
    </div>
  )
}

export default Auction