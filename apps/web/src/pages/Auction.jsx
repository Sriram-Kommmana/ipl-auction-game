import { useParams } from 'react-router-dom'
import { useAuctionStore } from '../store/auctionStore'
import PlayerCard from '../components/auction/PlayerCard'
import Timer from '../components/auction/Timer'
import CurrentBid from '../components/auction/CurrentBid'
import BidButton from '../components/auction/BidButton'
import ManagerControls from '../components/auction/ManagerControls'
import BidFeed from '../components/auction/BidFeed'
import PurseTracker from '../components/auction/PurseTracker'
import SquadViewer from '../components/auction/SquadViewer'

const Auction = ({ socketError, managerNotice }) => {
  const { roomId } = useParams()
  const lastResult = useAuctionStore((s) => s.lastResult)

  return (
    <div className="min-h-screen bg-mist px-4 py-6 sm:px-8 relative">
      {/* Widened to a real two-column dashboard now that BidFeed and
          PurseTracker exist to fill the sidebar. SquadViewer/ChatPanel
          will join the sidebar next. */}
      <div className="max-w-5xl mx-auto">
        <p className="text-xs uppercase tracking-wider text-ink/50 text-center mb-2">
          Room {roomId}
        </p>

        {managerNotice && (
          <div className="bg-brand-red text-paper text-sm text-center py-2 rounded-lg mb-4">
            {managerNotice.message}
          </div>
        )}

        <div className="flex justify-center mb-4">
          <Timer />
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="md:col-span-2 space-y-4">
            <PlayerCard />
            <CurrentBid />
            <BidButton />
            <ManagerControls />

            {socketError && (
              <p className="text-sm text-brand-red-dark text-center">{socketError.message}</p>
            )}
          </div>

          <div className="space-y-4">
            <BidFeed />
            <PurseTracker />
            <SquadViewer />
          </div>
        </div>
      </div>

      {lastResult && (
        <div className="fixed inset-0 bg-ink/80 flex items-center justify-center z-50">
          <div className="bg-paper rounded-2xl p-8 text-center max-w-sm mx-4">
            <p className={`font-display text-5xl ${
              lastResult.status === 'sold' ? 'text-brand-red' : 'text-ink/40'
            }`}>
              {lastResult.status === 'sold' ? 'SOLD!' : 'UNSOLD'}
            </p>
            <p className="font-display text-2xl text-ink mt-2">{lastResult.playerName}</p>
            {lastResult.status === 'sold' && (
              <p className="text-ink/60 mt-1">
                to {lastResult.teamName} for ₹{lastResult.soldFor}L
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default Auction