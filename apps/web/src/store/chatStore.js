import { create } from 'zustand'

export const useChatStore = create((set) => ({
  // ---- state ----
  messages: [], // [{ messageId, playerId, nickname, type, text, sentAt }]

  // ---- actions ----

  // Used on stateSync — restores chat history on reconnect
  setMessages: (messages) => set({ messages }),

  // Used on newChatMessage — append a single new message (user or broadcast)
  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message]
  })),

  clearMessages: () => set({ messages: [] })
}))