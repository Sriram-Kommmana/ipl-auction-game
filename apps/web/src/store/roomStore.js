import { create } from 'zustand'

export const useRoomStore = create((set) => ({
  // ---- state ----
  roomId: null,
  roomStatus: 'lobby',       // lobby | active | paused | completed
  auctionPhase: 'main',      // main | reauction
  players: [],               // [{ playerId, nickname, teamId, isManager, status, isBot }]
  teams: [],                 // [{ teamId, ownerId, purseLeft, playerCount, overseasCount }]
  pursePerTeam: 12500,
  managerPlayerId: null,
  maxPlayers: 25,
  maxOverseas: 8,

  // ---- actions ----

  // Used once on stateSync — replaces everything at once
  setRoomState: (roomData) => set({
    roomId: roomData.roomId,
    roomStatus: roomData.status,
    auctionPhase: roomData.auctionPhase,
    players: roomData.players,
    teams: roomData.teams,
    pursePerTeam: roomData.pursePerTeam,
    managerPlayerId: roomData.managerPlayerId,
    maxPlayers: roomData.maxPlayers,
    maxOverseas: roomData.maxOverseas
  }),

  setRoomStatus: (roomStatus) => set({ roomStatus }),

  // Used on teamSelected event — one player claims a team (or switches teams).
  // Updates BOTH players[] (their new teamId) and teams[] (ownerId claimed,
  // and previous team's ownerId cleared if they were switching teams).
  applyTeamSelection: (playerId, teamId, previousTeamId) => set((state) => ({
    players: state.players.map((p) =>
      p.playerId === playerId ? { ...p, teamId } : p
    ),
    teams: state.teams.map((t) => {
      if (t.teamId === teamId) return { ...t, ownerId: playerId }
      if (previousTeamId && t.teamId === previousTeamId) return { ...t, ownerId: null }
      return t
    })
  })),

  // Used on playerOnline/playerOffline for players ALREADY known locally
  // (e.g. a manager going offline briefly, then reconnecting).
  setPlayerStatus: (playerId, status) => set((state) => ({
    players: state.players.map((p) =>
      p.playerId === playerId ? { ...p, status } : p
    )
  })),

  // Used specifically on playerOnline — this is an UPSERT, not just an
  // update. Bug this fixes: joinRoom is a plain REST endpoint with no
  // Socket.IO access, so nothing broadcasts when a brand-new player joins.
  // The FIRST the room hears about them is playerOnline, fired once their
  // socket actually connects. If we only updated existing array entries
  // (like setPlayerStatus does), a genuinely new player's playerId would
  // match nothing in the array and silently vanish — which is exactly
  // what was happening until a manual refresh re-fetched the full list
  // via stateSync. New joiners always start with these exact defaults
  // per joinRoom.js's REST handler (teamId '', isManager false, isBot false).
  upsertPlayerOnline: (playerId, nickname) => set((state) => {
    const exists = state.players.some((p) => p.playerId === playerId)

    if (exists) {
      return {
        players: state.players.map((p) =>
          p.playerId === playerId ? { ...p, status: 'online' } : p
        )
      }
    }

    return {
      players: [
        ...state.players,
        { playerId, nickname, teamId: '', isManager: false, status: 'online', isBot: false }
      ]
    }
  }),

  // Used on playerSold event — updates the buying team's purse, squad count,
  // and overseas count (if applicable) all in one atomic state update.
  // isOverseas comes straight from the playerSold event payload (backend
  // includes it directly — no cross-store lookup needed).
  updateTeamAfterPurchase: (teamId, soldFor, isOverseas) => set((state) => ({
    teams: state.teams.map((t) =>
      t.teamId === teamId
        ? {
            ...t,
            purseLeft: t.purseLeft - soldFor,
            playerCount: t.playerCount + 1,
            overseasCount: isOverseas ? t.overseasCount + 1 : t.overseasCount
          }
        : t
    )
  }))
}))