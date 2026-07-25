// apps/web/src/hooks/useSession.js
import { useEffect, useState } from 'react'
import socket from '../lib/socket'
import { getSession, hasActiveSession } from '../lib/session'
import { useSessionStore } from '../store/sessionStore'

/**
 * Runs once on app load.
 * - If a valid session exists in localStorage, seeds sessionStore with it
 *   and connects the socket, emitting 'reconnect' once connected so the
 *   server can send back a stateSync.
 * - If no session exists, does nothing — App.jsx will render Home.
 *
 * Returns `hasSession` so App.jsx can decide whether to show Home
 * immediately, or wait for stateSync before routing to lobby/auction/results.
 *
 * NOTE: what happens if the server rejects the reconnect (e.g. room expired
 * or deleted) is NOT handled here — that's useSocket.js's responsibility,
 * via its 'reconnectError' listener, which should clear both localStorage
 * (lib/session.js clearSession) and sessionStore before routing to Home.
 */
export const useSession = () => {
  const setSession = useSessionStore((state) => state.setSession)
  const [hasSession] = useState(() => hasActiveSession())

  useEffect(() => {
    if (!hasSession) return

    const session = getSession()

    // teamId is obtained from the server during state synchronization.
    setSession({
      playerId: session.uuid,
      roomId: session.roomId,
      teamId: '',
      nickname: session.nickname,
      isManager: session.isManager
    })

    const emitReconnect = () => {
      socket.emit('reconnect', { playerId: session.uuid })
    }

    if (socket.connected) {
      emitReconnect()
    } else {
      socket.once('connect', emitReconnect)
      socket.connect()
    }

    return () => {
      socket.off('connect', emitReconnect)
    }
  }, [hasSession, setSession])

  return { hasSession }
}