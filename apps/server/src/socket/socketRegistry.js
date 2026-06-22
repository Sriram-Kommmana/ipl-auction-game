const socketToPlayer = new Map()
const playerToSocket = new Map()

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

const getPlayerIdBySocket = (socketId) => socketToPlayer.get(socketId)
const getSocketIdByPlayer = (playerId) => playerToSocket.get(playerId)

export {
    registerSocket,
    unregisterSocket,
    getPlayerIdBySocket,
    getSocketIdByPlayer
}