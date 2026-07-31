const SESSION_KEY = 'ipl-auction-session'

// Shape:
// {
//   uuid, roomId, nickname, playerPin, roomPin, isManager,
//   recents: [{ roomId, nickname, playerId, isManager, lastSeen }]
// }
//
// recents now stores playerId (the UUID assigned in that room) so a past
// room can be rejoined with ONE click on the SAME device — no PIN needed.
// PINs remain a fallback ONLY for recovering a session on a different
// device/browser where localStorage never had this data to begin with.

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

// Switches the "current" session to a room from recents — one click, no
// PIN. This only works because we're on the same device/browser that
// already holds that room's playerId. Note: playerPin/roomPin are NOT
// carried over for the newly-current room (recents never stored them,
// deliberately, to avoid stockpiling PINs in localStorage) — so if THIS
// device's storage gets wiped later, cross-device recovery for this
// specific room would need the PIN re-entered manually via JoinRoomForm.
// Acceptable tradeoff: that's a rare edge case, and the common path (same
// device, normal use) stays PIN-free as intended.
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