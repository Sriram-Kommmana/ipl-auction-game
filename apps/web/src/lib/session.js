const SESSION_KEY = 'ipl-auction-session'

// Shape:
// {
//   uuid, roomId, nickname, playerPin, roomPin, isManager,
//   recents: [{ roomId, nickname, playerId, isManager, lastSeen }]
// }

export const saveSession = ({ uuid, roomId, nickname, playerPin, roomPin, isManager }) => {
  const existing = getSession()

  const recents = existing?.recents || []
  const filteredRecents = recents.filter((r) => r.roomId !== roomId)
  filteredRecents.unshift({ roomId, nickname, playerId: uuid, isManager, lastSeen: Date.now() })

  const session = {
    uuid,
    roomId,
    nickname,
    playerPin,
    roomPin,
    isManager,
    recents: filteredRecents.slice(0, 5)
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

// BUG FIX: previously this did localStorage.removeItem(SESSION_KEY),
// which wiped the ENTIRE blob — including recents, which lives inside
// the same object. "Leave the current room" should not mean "forget every
// room I've ever played in". Now preserves recents, only clears identity.
export const clearSession = () => {
  const existing = getSession()
  if (existing?.recents?.length) {
    localStorage.setItem(SESSION_KEY, JSON.stringify({ recents: existing.recents }))
  } else {
    localStorage.removeItem(SESSION_KEY)
  }
}

export const hasActiveSession = () => {
  const session = getSession()
  return !!(session?.uuid && session?.roomId)
}

export const switchToRecentSession = (recent) => {
  const existing = getSession()
  const recents = existing?.recents || []

  const filteredRecents = recents.filter((r) => r.roomId !== recent.roomId)
  filteredRecents.unshift({ ...recent, lastSeen: Date.now() })

  const session = {
    uuid: recent.playerId,
    roomId: recent.roomId,
    nickname: recent.nickname,
    playerPin: '',
    roomPin: '',
    isManager: recent.isManager,
    recents: filteredRecents.slice(0, 5)
  }

  localStorage.setItem(SESSION_KEY, JSON.stringify(session))
  return session
}