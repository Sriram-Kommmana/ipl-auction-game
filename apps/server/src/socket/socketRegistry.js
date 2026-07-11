// Maps socket.id → playerId and playerId → socket.id
// Used to identify who disconnected and to find a player's current socket
const socketToPlayer = new Map()
const playerToSocket = new Map()

// Maps roomId → grace period setTimeout handle
// Private — only accessible via helper functions below
// Lives here (socket lifecycle module) not in timerManager (auction logic)
const managerGraceTimers = new Map()

const registerSocket = (socketId, playerId) => {
    // Clean up any stale mapping for this player first
    const oldSocketId = playerToSocket.get(playerId)
    if (oldSocketId && oldSocketId !== socketId) {
        socketToPlayer.delete(oldSocketId)
    }
    socketToPlayer.set(socketId, playerId)
    playerToSocket.set(playerId, socketId)
}

const unregisterSocket = (socketId) => {
    const playerId = socketToPlayer.get(socketId)
    if (playerId) {
        socketToPlayer.delete(socketId)
        playerToSocket.delete(playerId)
    }
    return playerId
}

const getPlayerIdBySocket  = (socketId) => socketToPlayer.get(socketId)
const getSocketIdByPlayer  = (playerId)  => playerToSocket.get(playerId)

// Grace timer helpers — called by onDisconnect and onReconnect
const startGraceTimer = (roomId, callback, delayMs) => {
    cancelGraceTimer(roomId)
    const handle = setTimeout(() => {
        managerGraceTimers.delete(roomId)
        callback()
    }, delayMs)
    managerGraceTimers.set(roomId, handle)
}

const cancelGraceTimer = (roomId) => {
    if (managerGraceTimers.has(roomId)) {
        clearTimeout(managerGraceTimers.get(roomId))
        managerGraceTimers.delete(roomId)
    }
}

const hasGraceTimer = (roomId) => managerGraceTimers.has(roomId)

export {
    registerSocket,
    unregisterSocket,
    getPlayerIdBySocket,
    getSocketIdByPlayer,
    startGraceTimer,
    cancelGraceTimer,
    hasGraceTimer
}