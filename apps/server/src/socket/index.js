import { onReconnect } from '../handlers/onReconnect.js'
import { onSelectTeam } from '../handlers/onSelectTeam.js'
import { onStartAuction } from '../handlers/onStartAuction.js'
import { onChat } from '../handlers/onChat.js'
import { onPause } from '../handlers/onPause.js'
import { onResume } from '../handlers/onResume.js'
import { onBid } from '../handlers/onBid.js'
import { onSkip } from '../handlers/onSkip.js'
import { onDisconnect } from '../handlers/onDisconnect.js'

const registerSocketHandlers = (io) => {
    io.on('connection', (socket) => {
        console.log(`[Socket] Connected: ${socket.id}`)

        socket.on('reconnect', (data) => onReconnect(io, socket, data))
        socket.on('selectTeam', (data) => onSelectTeam(io, socket, data))
        socket.on('startAuction', (data) => onStartAuction(io, socket, data))
        socket.on('sendChat', (data) => onChat(io, socket, data))
        socket.on('pauseAuction', (data) => onPause(io, socket, data))
        socket.on('resumeAuction', (data) => onResume(io, socket, data))
        socket.on('placeBid', (data) => onBid(io, socket, data))
        socket.on('skipPlayer', (data) => onSkip(io, socket, data))
        socket.on('disconnect', () => onDisconnect(io, socket))

        socket.on('disconnect', () => {
            console.log(`[Socket] Disconnected: ${socket.id}`)
        })
    })
}

export { registerSocketHandlers }