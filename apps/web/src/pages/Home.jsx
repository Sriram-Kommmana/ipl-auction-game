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

  const backgroundStyle = {
    background: `
      radial-gradient(
        circle at center,
        rgba(228, 38, 44, 0.12) 0%,
        rgba(228, 38, 44, 0.05) 30%,
        transparent 70%
      ),
      var(--color-mist)
    `,
  }

  if (linkedRoomId && matchingRecent && !showManualJoin) {
    return (
      <div
        className="relative min-h-screen overflow-hidden flex items-center justify-center px-4"
        style={backgroundStyle}
      >
        <div className="pointer-events-none absolute -top-40 -left-40 h-96 w-96 rounded-full bg-brand-red/7 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-40 -right-40 h-[28rem] w-[28rem] rounded-full bg-brand-red/7 blur-3xl" />
        <div className="pointer-events-none absolute top-1/4 -right-24 h-64 w-64 rounded-full bg-brand-red/5 blur-3xl" />

        <div className="relative z-10 w-full max-w-md text-center">
          <a
            href={import.meta.env.VITE_LANDING_URL}
            className="inline-block text-sm text-ink/50 hover:text-brand-red transition-colors mb-4"
          >
            ← Back to Home
          </a>

          <h1 className="font-display text-5xl tracking-wide leading-none text-ink mb-2">
            WELCOME <span className="text-brand-red">BACK</span>
          </h1>
          <p className="mt-3 mb-8 font-body text-sm text-ink/60">
            Continue as <span className="text-ink font-medium">{matchingRecent.nickname}</span> in
            room <span className="text-ink font-medium">{matchingRecent.roomId}</span>
          </p>

          <div className="overflow-hidden rounded-2xl border border-line bg-paper shadow-xl p-6 space-y-3">
            <button
              type="button"
              onClick={() => rejoinRecent(matchingRecent)}
              className="w-full bg-brand-red hover:bg-brand-red-dark text-paper font-display text-xl
                         tracking-wide py-3 rounded-lg transition-colors"
            >
              CONTINUE
            </button>
            <button
              type="button"
              onClick={() => setShowManualJoin(true)}
              className="text-sm text-ink/50 hover:text-ink transition-colors"
            >
              Not you? Join manually
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className="relative min-h-screen overflow-hidden flex items-center justify-center px-4"
      style={backgroundStyle}
    >
      {/* Background Blobs */}
      <div className="pointer-events-none absolute -top-40 -left-40 h-96 w-96 rounded-full bg-brand-red/7 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-40 -right-40 h-[28rem] w-[28rem] rounded-full bg-brand-red/7 blur-3xl" />
      <div className="pointer-events-none absolute top-1/4 -right-24 h-64 w-64 rounded-full bg-brand-red/5 blur-3xl" />

      {/* Content */}
      <div className="relative z-10 w-full max-w-lg">
        <a
          href={import.meta.env.VITE_LANDING_URL}
          className="inline-block text-sm text-ink/50 hover:text-brand-red transition-colors mb-4"
        >
          ← Back to Home
        </a>

        <div className="text-center mb-10">
          <h1 className="font-display text-6xl tracking-wide leading-none text-ink">
            CRICKET <span className="text-brand-red">AUCTION</span>
          </h1>

          <p className="mt-3 font-body text-sm text-ink/60">
            Build your dream IPL squad. No login. Just a room code and your
            friends.
          </p>
        </div>

        <div className="overflow-hidden rounded-2xl border border-line bg-paper shadow-xl">
          <div className="grid grid-cols-2">
            <button
              onClick={() => setMode('create')}
              className={`py-3 font-display text-xl tracking-wide transition-all duration-200 ${
                mode === 'create'
                  ? 'bg-brand-red text-paper'
                  : 'text-ink/50 hover:bg-mist hover:text-ink'
              }`}
            >
              CREATE ROOM
            </button>

            <button
              onClick={() => setMode('join')}
              className={`py-3 font-display text-xl tracking-wide transition-all duration-200 ${
                mode === 'join'
                  ? 'bg-brand-red text-paper'
                  : 'text-ink/50 hover:bg-mist hover:text-ink'
              }`}
            >
              JOIN ROOM
            </button>
          </div>

          <div className="p-8">
            {mode === 'create' ? <CreateRoomForm /> : <JoinRoomForm initialRoomId={linkedRoomId} />}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Home