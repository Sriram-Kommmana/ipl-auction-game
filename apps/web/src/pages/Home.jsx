import { useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import CreateRoomForm from '../components/home/CreateRoomForm'
import JoinRoomForm from '../components/home/JoinRoomForm'
import { getSession } from '../lib/session'
import { useRejoinRecent } from '../hooks/useRejoinRecent'

const Home = () => {
  const { roomId: linkedRoomId } = useParams()
  // Supports two distinct link types from the landing page:
  //   /join/:roomId    → pre-filled room code (share link, "Continue Playing")
  //   /?mode=join       → Join tab open, but empty (landing page's generic
  //                        "ENTER AUCTION" button — no specific room in mind)
  const [searchParams] = useSearchParams()
  const wantsJoinTab = Boolean(linkedRoomId) || searchParams.get('mode') === 'join'
  const [mode, setMode] = useState(wantsJoinTab ? 'join' : 'create')
  const [showManualJoin, setShowManualJoin] = useState(false)
  const rejoinRecent = useRejoinRecent()

  // If THIS device already has a recognized session for the linked room
  // (e.g. someone left mid-auction and is clicking their own share link
  // back), offer instant one-tap rejoin instead of making them retype
  // PIN + nickname like a stranger. Only checked once per mount — recents
  // don't change mid-session unless the user actually rejoins/switches.
  const matchingRecent = linkedRoomId
    ? getSession()?.recents?.find((r) => r.roomId === linkedRoomId.toUpperCase())
    : null

  if (linkedRoomId && matchingRecent && !showManualJoin) {
    return (
      <HomeShell>
        <div className="relative z-10 w-full max-w-md">
          <a href={import.meta.env.VITE_LANDING_URL} className="link-back inline-block mb-6">
            ← Back to Home
          </a>

          <p className="label-mono text-red mb-3">■ Session detected // 再接続</p>
          <h1 className="font-display text-7xl sm:text-8xl uppercase leading-[0.85] text-bone">
            Welcome<br />
            <span className="text-red glow-red">Back.</span>
          </h1>

          <div className="panel shadow-brutal mt-8">
            <div className="grid grid-cols-2 border-b border-line">
              <div className="p-4 border-r border-line">
                <p className="label-mono mb-1">Operator</p>
                <p className="font-display text-2xl text-bone truncate">{matchingRecent.nickname}</p>
              </div>
              <div className="p-4">
                <p className="label-mono mb-1">Room</p>
                <p className="font-display text-2xl text-bone tracking-[0.15em]">{matchingRecent.roomId}</p>
              </div>
            </div>
            <div className="p-5 space-y-4">
              <button
                type="button"
                onClick={() => rejoinRecent(matchingRecent)}
                className="btn-primary w-full text-2xl py-3"
              >
                Continue →
              </button>
              <button
                type="button"
                onClick={() => setShowManualJoin(true)}
                className="block mx-auto font-mono text-[11px] uppercase tracking-[0.18em] text-bone/45 hover:text-bone transition-colors"
              >
                Not you? Join manually
              </button>
            </div>
          </div>
        </div>
      </HomeShell>
    )
  }

  return (
    <HomeShell>
      <div className="relative z-10 w-full max-w-5xl grid lg:grid-cols-[1.1fr_1fr] gap-10 lg:gap-16 items-center">
        {/* ── Left: editorial masthead ── */}
        <div>
          <a href={import.meta.env.VITE_LANDING_URL} className="link-back inline-block mb-8">
            ← Back to Home
          </a>

          <div className="flex items-center gap-3 mb-4">
            <span className="h-2 w-2 bg-red animate-blink" />
            <p className="label-mono text-bone/60">Live multiplayer // 10 franchises // 1 purse</p>
          </div>

          <h1 className="font-display uppercase leading-[0.82] text-bone text-[4.5rem] sm:text-[7rem] lg:text-[8.5rem]">
            Cricket<br />
            <span className="text-red glow-red">Auction</span>
          </h1>

          <div className="mt-6 flex items-stretch gap-4 max-w-md">
            <div className="w-1 bg-red shrink-0" />
            <p className="text-base text-bone/65 leading-relaxed">
              Build your dream IPL squad. No login. Just a room code and your friends.
            </p>
          </div>

          <div className="hidden lg:grid grid-cols-3 mt-10 border-t border-line max-w-md">
            {[
              ['01', 'Create'],
              ['02', 'Claim team'],
              ['03', 'Outbid']
            ].map(([n, label]) => (
              <div key={n} className="pt-3 pr-3">
                <p className="font-mono text-[10px] text-red">{n}</p>
                <p className="font-display text-lg uppercase tracking-wide text-bone/80">{label}</p>
              </div>
            ))}
          </div>
        </div>

        {/* ── Right: form terminal ── */}
        <div className="panel shadow-brutal">
          <div className="flex items-center justify-between px-4 py-2 border-b border-line bg-void">
            <span className="label-mono">Terminal // 入札端末</span>
            <span className="flex gap-1.5">
              <span className="h-1.5 w-1.5 bg-red" />
              <span className="h-1.5 w-1.5 bg-bone/30" />
              <span className="h-1.5 w-1.5 bg-bone/30" />
            </span>
          </div>

          <div className="grid grid-cols-2 border-b border-line">
            <button
              onClick={() => setMode('create')}
              data-active={mode === 'create'}
              className="tab py-3.5 text-xl"
            >
              Create Room
            </button>
            <button
              onClick={() => setMode('join')}
              data-active={mode === 'join'}
              className="tab py-3.5 text-xl border-l border-line"
            >
              Join Room
            </button>
          </div>

          <div className="p-6 sm:p-8">
            {mode === 'create' ? <CreateRoomForm /> : <JoinRoomForm initialRoomId={linkedRoomId} />}
          </div>
        </div>
      </div>
    </HomeShell>
  )
}

// Shared backdrop for both Home states: grid + red horizon, plus a giant
// vertical katakana watermark and a slow scan beam.
const HomeShell = ({ children }) => (
  <div className="relative min-h-screen overflow-hidden flex items-center justify-center px-4 py-12 bg-city">
    <div
      aria-hidden
      className="pointer-events-none select-none absolute right-2 sm:right-6 top-1/2 -translate-y-1/2
                 font-jp font-black text-[5.5rem] sm:text-[9rem] leading-none text-bone/[0.035]
                 [writing-mode:vertical-rl]"
    >
      オークション
    </div>
    <div
      aria-hidden
      className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-red/0 via-red/[0.05] to-red/0 animate-scan"
    />
    <div aria-hidden className="hidden sm:block pointer-events-none absolute left-0 top-0 h-full w-px bg-red/40 ml-8" />
    {children}
  </div>
)

export default Home