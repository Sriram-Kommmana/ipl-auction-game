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
    <div className="min-h-screen md:h-screen md:overflow-hidden bg-city px-4 py-4 sm:px-8 relative">
      {/* ================= MOBILE (below md) ================= */}
      <div className="md:hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <button
            type="button"
            onClick={() => setShowLeaveConfirm(true)}
            className="link-back whitespace-nowrap"
          >
            ← Leave
          </button>
          <RoomCode roomId={roomId} size="sm" />
          <ManagerControls />
        </div>

        {managerNotice && (
          <div className="flex items-stretch border-2 border-red bg-red/10 mb-4">
            <div className="hazard w-3 shrink-0" />
            <p className="flex-1 font-mono text-xs uppercase tracking-wider text-bone text-center py-2 px-3">
              {managerNotice.message}
            </p>
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
          className="fixed bottom-5 right-5 z-30 bg-red text-bone border-2 border-red p-4
                     shadow-[4px_4px_0_0_var(--color-bone)] active:translate-x-1 active:translate-y-1 active:shadow-none
                     transition-transform flex items-center justify-center"
          aria-label="Open Chat, Bids, Purse & Squad"
        >
          <Menu size={22} />
        </button>

        <Drawer.Root open={isMobileDrawerOpen} onOpenChange={setIsMobileDrawerOpen}>
          <Drawer.Portal>
            <Drawer.Overlay className="fixed inset-0 bg-void/80 backdrop-blur-sm z-40" />
            <Drawer.Content className="fixed bottom-0 left-0 right-0 bg-panel border-t-2 border-red z-50 flex flex-col h-[75vh] outline-none">
              <div className="mx-auto mt-2.5 h-1 w-12 bg-bone/25 shrink-0" />

              <div className="flex border-b border-line shrink-0 mt-2">
                {MOBILE_TABS.map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setMobileTab(tab.id)}
                    data-active={mobileTab === tab.id}
                    className="tab flex-1 text-sm py-2.5"
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
            className="link-back"
          >
            ← Leave Room
          </button>

          <RoomCode roomId={roomId} size="sm" />

          <ManagerControls />
        </div>

        {managerNotice && (
          <div className="flex items-stretch border-2 border-red bg-red/10 mb-4 md:shrink-0">
            <div className="hazard w-3 shrink-0" />
            <p className="flex-1 font-mono text-xs uppercase tracking-wider text-bone text-center py-2 px-3">
              {managerNotice.message}
            </p>
            <div className="hazard w-3 shrink-0" />
          </div>
        )}

        <div className="grid md:grid-cols-3 gap-4 md:flex-1 md:min-h-0">
          <div className="flex flex-col gap-4 md:min-h-0">
            {/* 50/50 split — each panel fills its half (h-full inside) and
                scrolls its own list, so there's no dead gap between them. */}
            <div className="md:flex-1 md:min-h-0 md:overflow-hidden">
              <PurseTracker />
            </div>
            <div className="md:flex-1 md:min-h-0 md:overflow-hidden">
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

          <div className="md:flex md:flex-col md:min-h-0 panel overflow-hidden">
            <div className="flex border-b border-line shrink-0">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  data-active={activeTab === tab.id}
                  className="tab flex-1 text-sm py-2.5"
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
            className="fixed inset-0 bg-void/85 backdrop-blur-sm flex items-center justify-center z-50"
          >
            <motion.div
              initial={{ scale: 0.6, opacity: 0, rotate: -4 }}
              animate={{ scale: 1, opacity: 1, rotate: 0 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className={`relative bg-panel border-2 text-center max-w-sm w-full mx-4 overflow-hidden
                ${lastResult.status === 'sold'
                  ? 'border-red shadow-[8px_8px_0_0_var(--color-red)]'
                  : 'border-line-strong shadow-[8px_8px_0_0_var(--color-line-strong)]'}`}
            >
              <div className={`h-3 ${lastResult.status === 'sold' ? 'hazard' : 'bg-line-strong'}`} />
              <div className="px-8 pt-6 pb-8">
                <p className="font-jp font-black text-sm tracking-[0.5em] text-bone/40">
                  {lastResult.status === 'sold' ? '落札' : '不落札'}
                </p>
                <p className={`font-display text-7xl uppercase leading-none mt-1 ${
                  lastResult.status === 'sold' ? 'text-red glow-red' : 'text-bone/35'
                }`}>
                  {lastResult.status === 'sold' ? 'Sold!' : 'Unsold'}
                </p>
                <div className="h-px bg-line my-4" />
                <p className="font-display text-3xl uppercase tracking-wide text-bone">{lastResult.playerName}</p>
                {lastResult.status === 'sold' && (
                  <p className="font-mono text-xs uppercase tracking-wider text-bone/55 mt-2">
                    to <span className="text-bone">{lastResult.teamName}</span> for{' '}
                    <span className="text-red">₹{lastResult.soldFor}L</span>
                  </p>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default Auction