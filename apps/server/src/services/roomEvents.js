import { EventEmitter } from 'node:events'

// In-process feed of auction lifecycle events.
//
// Every lifecycle broadcast goes through emitToRoom, which sends it to the
// room's sockets AND publishes it here. Server-side bots subscribe to this
// instead of pretending to be socket clients, so they observe exactly what
// humans see, in the same order, without any socket plumbing.
//
// Listeners must not throw or block: the engine emits synchronously and
// carries on. botManager defers its own work to the next tick.
const roomEvents = new EventEmitter()
roomEvents.setMaxListeners(20)

const emitToRoom = (io, roomId, event, payload) => {
    io.to(roomId).emit(event, payload)
    try {
        roomEvents.emit('event', { roomId, event, payload })
    } catch (err) {
        console.error(`[roomEvents] listener failed for ${event} in room ${roomId}:`, err)
    }
}

export { roomEvents, emitToRoom }
