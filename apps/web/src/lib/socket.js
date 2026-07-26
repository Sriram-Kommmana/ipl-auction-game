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

export default socket