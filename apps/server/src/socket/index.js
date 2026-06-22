import { onReconnect } from '../handlers/onReconnect.js'

const registerSocketHandlers = (io) => {
    io.on('connection', (socket) => {
        console.log(`[Socket] Connected: ${socket.id}`)

        socket.on('reconnect', (data) => onReconnect(io, socket, data))

        socket.on('disconnect', () => {
            console.log(`[Socket] Disconnected: ${socket.id}`)
        })
    })
}

export { registerSocketHandlers }