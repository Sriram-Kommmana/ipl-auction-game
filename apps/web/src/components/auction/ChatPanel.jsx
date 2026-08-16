import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useChatStore } from '../../store/chatStore'
import { useSessionStore } from '../../store/sessionStore'
import { useSocketConnected } from '../../hooks/useSocketConnected'
import socket from '../../lib/socket'

const ChatPanel = () => {
  const messages = useChatStore((s) => s.messages)
  const playerId = useSessionStore((s) => s.playerId)
  const isConnected = useSocketConnected()

  const [text, setText] = useState('')
  const scrollRef = useRef(null)
  const bottomRef = useRef(null)
  const isNearBottomRef = useRef(true)

  const handleScroll = () => {
    const el = scrollRef.current
    if (!el) return
    const threshold = 80
    isNearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < threshold
  }

  // Compute the REAL initial scroll position before first paint, rather
  // than assuming "at bottom" — a fresh stateSync could load 200 old
  // messages, and we shouldn't force-scroll based on an unverified guess.
  useLayoutEffect(() => {
    handleScroll()
  }, [])

  useEffect(() => {
    if (!isNearBottomRef.current) return
    // Smooth for the first few messages (feels nice, low volume). Once a
    // room has real chat history, switch to instant — animating a scroll
    // on every message during a burst of activity feels sluggish rather
    // than polished.
    const behavior = messages.length < 5 ? 'smooth' : 'auto'
    bottomRef.current?.scrollIntoView({ behavior })
  }, [messages])

  const handleSend = (e) => {
    e.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || !isConnected) return
    socket.emit('sendChat', { playerId, text: trimmed })
    setText('')
  }

  return (
    <div className="bg-paper border border-line rounded-2xl p-4 h-full flex flex-col">
      <h2 className="font-display text-lg text-ink mb-2 tracking-wide">CHAT</h2>

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="space-y-1.5 flex-1 min-h-0 overflow-y-auto mb-3"
      >
        {messages.length === 0 && (
          <p className="text-xs text-ink/30 text-center py-4">Be the first to say something.</p>
        )}
        {messages.map((msg) =>
          msg.type === 'broadcast' ? (
            <p key={msg.messageId} className="text-[11px] text-ink/40 text-center italic py-1">
              {msg.text}
            </p>
          ) : (
            <div key={msg.messageId} className="text-sm">
              <span className="text-ink/50 text-xs">{msg.nickname}: </span>
              <span className="text-ink">{msg.text}</span>
            </div>
          )
        )}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSend} className="flex gap-2">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={isConnected ? 'Type a message…' : 'Reconnecting…'}
          disabled={!isConnected}
          maxLength={200}
          className="flex-1 bg-mist border border-line rounded-lg px-3 py-1.5 text-sm text-ink
                     placeholder:text-ink/30 focus:outline-none focus:ring-2 focus:ring-brand-red
                     disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!isConnected || !text.trim()}
          className="bg-brand-red hover:bg-brand-red-dark disabled:opacity-40 disabled:cursor-not-allowed
                     text-paper font-display text-sm px-4 rounded-lg transition-colors"
        >
          SEND
        </button>
      </form>
    </div>
  )
}

export default ChatPanel