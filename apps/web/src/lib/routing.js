// Maps a room's backend status to the correct frontend route. Used by
// both App.jsx (reopened-tab / stateSync redirect) and JoinRoomForm.jsx
// (rejoining a room that's already mid-auction) so this decision is made
// the same way in exactly one place, not duplicated and allowed to drift.
export const getRoomRoute = (roomStatus, roomId) => {
  if (roomStatus === 'lobby') return `/lobby/${roomId}`
  if (roomStatus === 'active' || roomStatus === 'paused') return `/auction/${roomId}`
  if (roomStatus === 'completed') return `/results/${roomId}`
  return '/'
}