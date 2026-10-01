import { getSession } from './session'

// Which team on a results page is "yours": the team whose owner is the
// player id this browser joined the room with — the active session if it is
// this room, otherwise the room's entry in the recent-rooms list. Null for
// spectators (e.g. someone opening a shared results link).
export const findMyTeamId = (teams, roomId) => {
  const session = getSession()
  const playerId = session?.roomId === roomId
    ? session.uuid
    : session?.recents?.find((r) => r.roomId === roomId)?.playerId
  if (!playerId) return null
  return teams.find((t) => t.ownerId === playerId)?.teamId ?? null
}

export const ordinal = (n) => {
  const tens = n % 100
  if (tens >= 11 && tens <= 13) return `${n}th`
  return `${n}${{ 1: 'st', 2: 'nd', 3: 'rd' }[n % 10] ?? 'th'}`
}

// "01", or "=03" for a shared rank.
export const rankLabel = ({ rank, tied }) => `${tied ? '=' : ''}${String(rank).padStart(2, '0')}`

export const TIEBREAK_LABELS = {
  xiStrength: 'XI strength',
  injuryCover: 'injury cover',
  purseSpent: 'less purse spent'
}
