import redis from '../redis/client.js'
import { IPL_TEAMS } from '../constants.js'
import { claimTeam } from '../services/claimTeam.js'

const onSelectTeam = async (io, socket, data) => {
    const { playerId, teamId } = data || {}

    if (!playerId) {
        return socket.emit('selectTeamError', { message: 'playerId is required' })
    }

    if (!teamId || !IPL_TEAMS[teamId]) {
        return socket.emit('selectTeamError', { message: 'Invalid team selected' })
    }

    const session = await redis.hgetall(`session:${playerId}`)
    if (!session || !session.roomId) {
        return socket.emit('selectTeamError', { message: 'Session not found. Please rejoin the room.' })
    }

    const { roomId } = session
    const result = await claimTeam({ roomId, playerId, teamId })

    if (!result.ok) {
        return socket.emit('selectTeamError', { message: result.error })
    }

    // Already on this team — nothing to do, just re-confirm
    if (result.unchanged) {
        return socket.emit('teamSelected', {
            playerId,
            teamId,
            teamName: IPL_TEAMS[teamId]
        })
    }

    // Broadcast to everyone — frontend updates the team grid for all
    io.to(roomId).emit('teamSelected', {
        playerId,
        teamId,
        teamName:       IPL_TEAMS[teamId],
        previousTeamId: result.previousTeamId,
        totalTeams:     result.totalTeams
    })
}

export { onSelectTeam, IPL_TEAMS }
