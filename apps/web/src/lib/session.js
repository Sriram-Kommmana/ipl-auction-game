const SESSION_KEY = 'ipl-auction-session'

// Shape:
// {
//   uuid, roomId, nickname, playerPin, roomPin, isManager,
//   recents: [{ roomId, nickname, lastSeen }]
// }

export const saveSession = ({ uuid, roomId, nickname, playerPin, roomPin, isManager }) => {
  const existing = getSession()

  const recents = existing?.recents || []
  const filteredRecents = recents.filter((r) => r.roomId !== roomId)
  filteredRecents.unshift({ roomId, nickname, lastSeen: Date.now() })

  const session = {
    uuid,
    roomId,
    nickname,
    playerPin,
    roomPin,
    isManager,
    recents: filteredRecents.slice(0, 5) // keep last 5 rooms
  }

  localStorage.setItem(SESSION_KEY, JSON.stringify(session))
  return session
}

export const getSession = () => {
  try {
    const raw = localStorage.getItem(SESSION_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export const clearSession = () => {
  localStorage.removeItem(SESSION_KEY)
}

export const hasActiveSession = () => {
  const session = getSession()
  return !!(session?.uuid && session?.roomId)
}