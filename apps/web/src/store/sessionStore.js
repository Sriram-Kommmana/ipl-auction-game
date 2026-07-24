import { create } from 'zustand'

export const useSessionStore = create((set) => ({
  // ---- state ----
  playerId: null,
  roomId: null,
  teamId: '',
  nickname: '',
  isManager: false,

  // ---- actions (functions that update state) ----
  setSession: (session) => set({
    playerId: session.playerId,
    roomId: session.roomId,
    teamId: session.teamId,
    nickname: session.nickname,
    isManager: session.isManager
  }),

  setTeamId: (teamId) => set({ teamId }),

  clearSession: () => set({
    playerId: null,
    roomId: null,
    teamId: '',
    nickname: '',
    isManager: false
  })
}))