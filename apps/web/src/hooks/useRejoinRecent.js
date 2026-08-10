import { useNavigate } from 'react-router-dom'
import { switchToRecentSession } from '../lib/session'
import { connectAndReconnect } from '../lib/socket'
import { useSessionStore } from '../store/sessionStore'

/**
 * One-tap rejoin — no PIN, no REST call. Used by both JoinRoomForm's
 * "Continue Playing" cards AND Home.jsx's /join/:roomId flow, when this
 * device already recognizes the room from a previous session.
 */
export const useRejoinRecent = () => {
  const navigate = useNavigate()
  const setSession = useSessionStore((s) => s.setSession)

  return (recent) => {
    switchToRecentSession(recent)
    setSession({
      playerId: recent.playerId,
      roomId: recent.roomId,
      teamId: '',
      nickname: recent.nickname,
      isManager: recent.isManager
    })
    connectAndReconnect(recent.playerId)
    navigate('/')
  }
}