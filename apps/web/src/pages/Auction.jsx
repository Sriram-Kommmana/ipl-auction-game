import { useState } from 'react'
import { Drawer } from 'vaul'
import { Menu } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import { useParams } from 'react-router-dom'
import { useAuctionStore } from '../store/auctionStore'
import { useLeaveRoom } from '../hooks/useLeaveRoom'
import PlayerCard from '../components/auction/PlayerCard'
import Timer from '../components/auction/Timer'
import CurrentBid from '../components/auction/CurrentBid'
import BidButton from '../components/auction/BidButton'
import ManagerControls from '../components/auction/ManagerControls'
import BidFeed from '../components/auction/BidFeed'
import PurseTracker from '../components/auction/PurseTracker'
import SquadViewer from '../components/auction/SquadViewer'
import ChatPanel from '../components/auction/ChatPanel'
import RoomCode from '../components/shared/RoomCode'
import Modal from '../components/shared/Modal'

// Desktop only — BidFeed/PurseTracker are standalone panels there, not tabs
const TABS = [
  { id: 'chat', label: 'Chat', Component: ChatPanel },
  { id: 'squadViewer', label: 'My Squad', Component: SquadViewer }
]

// Mobile only — all 4 secondary panels live as tabs inside the drawer
const MOBILE_TABS = [
  { id: 'bidFeed', label: 'Bids', Component: BidFeed },
  { id: 'purseTracker', label: 'Purse', Component: PurseTracker },
  { id: 'chat', label: 'Chat', Component: ChatPanel },
  { id: 'squadViewer', label: 'Squad', Component: SquadViewer }
]

const Auction = ({ managerNotice }) => {
  const { roomId } = useParams()
  const lastResult = useAuctionStore((s) => s.lastResult)
  const leaveRoom = useLeaveRoom()

  const [activeTab, setActiveTab] = useState('chat')
  const [mobileTab, setMobileTab] = useState('chat')
  const [isMobileDrawerOpen, setIsMobileDrawerOpen] = useState(false)
  const [showLeaveConfirm, setShowLeaveConfirm] = useState(false)

  const ActiveTabComponent = TABS.find((t) => t.id === activeTab)?.Component
  const MobileActiveComponent = MOBILE_TABS.find((t) => t.id === mobileTab)?.Component

  return (
    <div className="min-h-screen md:h-screen md:overflow-hidden bg-mist px-4 py-4 sm:px-8 relative">
      {/* ================= MOBILE (below md) ================= */}
      <div className="md:hidden">
        <div className="flex items-center justify-between mb-4">
          <button
            type="button"
            onClick={() => setShowLeaveConfirm(true)}
            className="text-sm text-ink/50 hover:text-brand-red transition-colors"
          >
            ← Leave Room
          </button>
          <RoomCode roomId={roomId} size="sm" />
          <ManagerControls />
        </div>

        {managerNotice && (
          <div className="bg-brand-red text-paper text-sm text-center py-2 rounded-lg mb-4">
            {managerNotice.message}
          </div>
        )}

        <div className="px-2 py-1">
          <PlayerCard />
        </div>

        <div className="grid grid-cols-[minmax(0,1fr)_92px] gap-2 items-start mt-3">
          <BidButton />
          <div className="h-16">
            <Timer variant="compact" />
          </div>
        </div>

        <div className="mt-3">
          <CurrentBid />
        </div>

        {/* Floating button — the ONLY thing that opens the drawer. Being
            md:hidden here works correctly since this is a plain DOM
            element, not portaled content. */}
        <button
          type="button"
          onClick={() => setIsMobileDrawerOpen(true)}
          className="fixed bottom-5 right-5 z-30 bg-brand-red text-paper rounded-full p-4 shadow-lg
                     flex items-center justify-center"
          aria-label="Open Chat, Bids, Purse & Squad"
        >
          <Menu size={22} />
        </button>

        <Drawer.Root open={isMobileDrawerOpen} onOpenChange={setIsMobileDrawerOpen}>
          <Drawer.Portal>
            <Drawer.Overlay className="fixed inset-0 bg-ink/60 z-40" />
            <Drawer.Content className="fixed bottom-0 left-0 right-0 bg-paper rounded-t-2xl border-t border-line z-50 flex flex-col h-[75vh]">
              <div className="mx-auto mt-2 h-1.5 w-10 rounded-full bg-ink/20 shrink-0" />

              <div className="flex border-b border-line shrink-0 mt-2">
                {MOBILE_TABS.map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setMobileTab(tab.id)}
                    className={`flex-1 text-xs font-display tracking-wide py-2.5 transition-colors
                      ${mobileTab === tab.id
                        ? 'bg-brand-red text-paper'
                        : 'text-ink/50 hover:text-ink hover:bg-mist'}`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="p-3 overflow-y-auto flex-1">
                {MobileActiveComponent && <MobileActiveComponent />}
              </div>
            </Drawer.Content>
          </Drawer.Portal>
        </Drawer.Root>
      </div>

      {/* ================= DESKTOP (md and up) — unchanged ================= */}
      <div className="hidden md:flex md:flex-col md:h-full md:min-h-0 max-w-6xl w-full mx-auto">
        <div className="flex items-center justify-between mb-4 md:shrink-0">
          <button
            type="button"
            onClick={() => setShowLeaveConfirm(true)}
            className="text-sm text-ink/50 hover:text-brand-red transition-colors"
          >
            ← Leave Room
          </button>

          <RoomCode roomId={roomId} size="sm" />

          <ManagerControls />
        </div>

        {managerNotice && (
          <div className="bg-brand-red text-paper text-sm text-center py-2 rounded-lg mb-4 md:shrink-0">
            {managerNotice.message}
          </div>
        )}

        <div className="grid md:grid-cols-3 gap-4 md:flex-1 md:min-h-0">
          <div className="flex flex-col gap-4 md:min-h-0">
            <div className="md:flex-1 md:min-h-0 md:overflow-y-auto">
              <PurseTracker />
            </div>
            <div className="md:flex-1 md:min-h-0 md:overflow-y-auto">
              <BidFeed />
            </div>
          </div>

          <div
            className="space-y-3 md:overflow-y-auto md:overflow-x-hidden md:min-h-0"
            style={{ scrollbarGutter: 'stable' }}
          >
            <div className="px-4 py-1">
              <PlayerCard />
            </div>

            <div className="grid grid-cols-[minmax(0,1fr)_92px] gap-2 items-start">
              <BidButton />
              <div className="h-16">
                <Timer variant="compact" />
              </div>
            </div>

            <CurrentBid />
          </div>

          <div className="md:flex md:flex-col md:min-h-0 bg-paper border border-line rounded-2xl overflow-hidden">
            <div className="flex border-b border-line shrink-0">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 text-xs font-display tracking-wide py-2.5 transition-colors
                    ${activeTab === tab.id
                      ? 'bg-brand-red text-paper'
                      : 'text-ink/50 hover:text-ink hover:bg-mist'}`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            <div className="p-3 md:overflow-y-auto md:min-h-0 flex-1">
              {ActiveTabComponent && <ActiveTabComponent />}
            </div>
          </div>
        </div>
      </div>

      <Modal
        isOpen={showLeaveConfirm}
        title="Leave this room?"
        message="You'll be able to rejoin from the Home screen's Continue Playing list."
        confirmLabel="Leave"
        danger
        onConfirm={leaveRoom}
        onCancel={() => setShowLeaveConfirm(false)}
      />

      <AnimatePresence>
        {lastResult && (
          <motion.div
            key="result-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-ink/80 flex items-center justify-center z-50"
          >
            <motion.div
              initial={{ scale: 0.6, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className="bg-paper rounded-2xl p-8 text-center max-w-sm mx-4"
            >
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
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default Auction