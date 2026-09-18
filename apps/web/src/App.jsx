import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { useSession } from './hooks/useSession'
import { useSocket } from './hooks/useSocket'
import { useRoomStore } from './store/roomStore'
import { getRoomRoute } from './lib/routing'
import Toast from './components/shared/Toast'
import Home from './pages/Home'
import Lobby from './pages/Lobby'
import Auction from './pages/Auction'
import PostAuction from './pages/PostAuction'

// Separate from App itself because useSocket()/useNavigate() require being
// INSIDE <BrowserRouter> — can't call them in the component that renders it.
const AppRoutes = () => {
  // Still called for its side effect (seeds sessionStore + connects socket
  // if there's a session at initial page load) — its returned hasSession
  // boolean is no longer used here, since it goes stale after mount (see
  // note below).
  useSession()
  // managerNotice stays prop-drilled to Auction — it's a persistent,
  // Auction-specific banner (different purpose/styling than a transient
  // dismissible toast). socketError now feeds the global Toast instead
  // of being drilled through routes.
  const { managerNotice, socketError, clearSocketError } = useSocket()

  const navigate = useNavigate()
  const location = useLocation()

  const roomId = useRoomStore((s) => s.roomId)
  const roomStatus = useRoomStore((s) => s.roomStatus)

  // Handles the "closed the tab, reopened the site fresh" case: localStorage
  // says there's a session, stateSync eventually confirms the room's real
  // status, and if we're still sitting at "/" (not already on a room-
  // specific route), send the user to wherever the room actually is.
  // hasSession from useSession() is a ONE-TIME snapshot from initial mount
  // — it goes stale the moment a "Continue Playing" card establishes a NEW
  // session mid-session (no page reload). roomId/roomStatus are already
  // live Zustand state and naturally stay null for a genuinely fresh
  // visitor, so they're sufficient on their own — no need for the extra
  // (and buggy) hasSession gate.
  useEffect(() => {
    if (!roomId || !roomStatus) return
    if (location.pathname !== '/') return

    navigate(getRoomRoute(roomStatus, roomId))
  }, [roomId, roomStatus, location.pathname, navigate])

  return (
    <>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/join/:roomId" element={<Home />} />
        <Route path="/lobby/:roomId" element={<Lobby />} />
        <Route path="/auction/:roomId" element={<Auction managerNotice={managerNotice} />} />
        <Route path="/results/:roomId" element={<PostAuction />} />
      </Routes>
      <Toast message={socketError?.message} onDismiss={clearSocketError} />
      {/* Global CRT scanline veil — purely decorative, never intercepts clicks */}
      <div aria-hidden className="pointer-events-none fixed inset-0 z-[100] scanlines opacity-50" />
    </>
  )
}

const App = () => (
  <BrowserRouter>
    <AppRoutes />
  </BrowserRouter>
)

export default App