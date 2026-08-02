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
  // This is an UPSERT, not just an update: teams[] starts EMPTY in a fresh
  // room (room:{roomId}:teams only gets an entry once a team is first
  // claimed), so the very first claim of any team won't match anything via
  // .map() alone — same bug class as playerOnline needing upsertPlayerOnline.
  // teamName comes straight from the event so a brand-new entry can be built
  // with the same shape as stateSync's teams[] (purseLeft/playerCount/etc.
  // start at their room defaults since nobody's bought anything yet).
  applyTeamSelection: (playerId, teamId, previousTeamId, teamName) => set((state) => {
    const teamExists = state.teams.some((t) => t.teamId === teamId)

    const clearPrevious = (list) =>
      previousTeamId
        ? list.map((t) => (t.teamId === previousTeamId ? { ...t, ownerId: null } : t))
        : list

    const teams = teamExists
      ? clearPrevious(state.teams).map((t) =>
          t.teamId === teamId ? { ...t, ownerId: playerId } : t
        )
      : [
          ...clearPrevious(state.teams),
          {
            teamId,
            name: teamName,
            ownerId: playerId,
            purseLeft: state.pursePerTeam,
            purseSpent: 0,
            playerCount: 0,
            overseasCount: 0,
            isBot: false,
            squad: []
          }
        ]

    return {
      players: state.players.map((p) =>
        p.playerId === playerId ? { ...p, teamId } : p
      ),
      teams
    }
  }),

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
  })),

  // Used when leaving a room — without this, Zustand's global store would
  // keep stale data from the abandoned room (e.g. browser back button into
  // an old /lobby/:roomId would render off leftover state, never cleared).
  resetRoom: () => set({
    roomId: null,
    roomStatus: 'lobby',
    auctionPhase: 'main',
    players: [],
    teams: [],
    pursePerTeam: 12500,
    managerPlayerId: null,
    maxPlayers: 25,
    maxOverseas: 8
  })
}))