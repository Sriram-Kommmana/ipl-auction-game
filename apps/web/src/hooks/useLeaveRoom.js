// apps/web/src/hooks/useLeaveRoom.js
import { useNavigate } from 'react-router-dom'
import { useSessionStore } from '../store/sessionStore'
import { useRoomStore } from '../store/roomStore'
import { useAuctionStore } from '../store/auctionStore'
import { useChatStore } from '../store/chatStore'
import socket from '../lib/socket'
import { clearSession } from '../lib/session'

/**
 * Shared "leave the room" logic — used by both LobbyControls and
 * ManagerControls. Clears localStorage, all 4 Zustand stores, disconnects
 * the socket, and returns to Home.
 */
export const useLeaveRoom = () => {
  const navigate = useNavigate()
  const playerId = useSessionStore((s) => s.playerId)
  const clearSessionStore = useSessionStore((s) => s.clearSession)
  const resetRoom = useRoomStore((s) => s.resetRoom)
  const resetAuction = useAuctionStore((s) => s.resetAuction)
  const clearMessages = useChatStore((s) => s.clearMessages)

  return () => {
    socket.emit('leaveRoom', { playerId })

    clearSession()
    clearSessionStore()
    resetRoom()
    resetAuction()
    clearMessages()
    navigate('/')

    // socket.js deliberately leaves `transports` unspecified (Cloudflare/
    // Hetzner compatibility — see socket.js notes), meaning leaveRoom could
    // be sent over HTTP long-polling rather than a WebSocket. A WebSocket's
    // send() generally flushes before close() even with no delay, but an
    // in-flight POLLING request is a real network round-trip that an
    // immediate disconnect() can abort outright. A 0ms setTimeout wouldn't
    // help here — it only defers to the next JS tick, not real network
    // time — so this uses an actual small delay instead.
    setTimeout(() => {
      socket.disconnect()
    }, 150)
  }
}