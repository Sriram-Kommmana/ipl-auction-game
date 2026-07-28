// apps/web/src/lib/socket.js
import { io } from 'socket.io-client'

// transports intentionally left unspecified (not forced to ['websocket']):
// production target is Hetzner VPS behind Cloudflare SSL, and Socket.IO's
// default polling-then-upgrade behavior is what gracefully handles a
// reverse-proxy layer if WebSocket upgrade isn't perfectly configured there.
// Revisit once WS upgrade is confirmed working end-to-end in production.
const socket = io(import.meta.env.VITE_SERVER_URL, {
  autoConnect: false,
  reconnection: true,
  reconnectionAttempts: Infinity,
  reconnectionDelay: 1000,
  timeout: 10000
})

if (import.meta.env.DEV) {
  socket.on('connect', () => console.debug('[socket] connected', socket.id))
  socket.on('disconnect', (reason) => console.debug('[socket] disconnected:', reason))
  socket.on('connect_error', (err) => console.debug('[socket] connect_error:', err.message))
}

// Shared by useSession.js (on app load, from localStorage) AND
// CreateRoomForm.jsx/JoinRoomForm.jsx (right after a successful REST call,
// no page reload involved) — both need the exact same "connect, then once
// actually connected, emit reconnect" sequence to get the server's
// stateSync. Centralized here so that sequencing logic exists in one place.
//
// pendingReconnectListener tracks an in-flight call so this stays
// idempotent: if connectAndReconnect is called again before the socket
// has finished connecting (e.g. a rapid double-submit), the previous
// call's listener is removed before registering the new one — otherwise
// two separate 'connect' listeners would both fire once connected,
// emitting 'reconnect' twice.
let pendingReconnectListener = null

export const connectAndReconnect = (playerId) => {
  const emitReconnect = () => {
    pendingReconnectListener = null
    socket.emit('reconnect', { playerId })
  }

  if (socket.connected) {
    emitReconnect()
    return
  }

  if (pendingReconnectListener) {
    socket.off('connect', pendingReconnectListener)
  }

  pendingReconnectListener = emitReconnect
  socket.once('connect', emitReconnect)
  socket.connect()
}

export default socket