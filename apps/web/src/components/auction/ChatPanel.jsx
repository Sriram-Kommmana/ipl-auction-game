import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'
import { useChatStore } from '../../store/chatStore'
import { useRoomStore } from '../../store/roomStore'
import { useSessionStore } from '../../store/sessionStore'
import { useSocketConnected } from '../../hooks/useSocketConnected'
import socket from '../../lib/socket'

const ChatPanel = () => {
  const messages = useChatStore((s) => s.messages)
  const playerId = useSessionStore((s) => s.playerId)
  const players = useRoomStore((s) => s.players)
  const isConnected = useSocketConnected()

  // Sender's CURRENT team, looked up live by playerId (chat messages carry
  // playerId, not teamId) — so a name tag always shows team colors, even
  // for messages sent before that player picked a team.
  const teamByPlayerId = useMemo(
    () => Object.fromEntries(players.map((p) => [p.playerId, p.teamId ? TEAMS_BY_ID[p.teamId] : null])),
    [players]
  )

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
    <div className="h-full flex flex-col">
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="space-y-1.5 flex-1 min-h-0 overflow-y-auto mb-3"
      >
        {messages.length === 0 && (
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-bone/30 text-center py-4">Be the first to say something.</p>
        )}
        {messages.map((msg) =>
          msg.type === 'broadcast' ? (
            <p key={msg.messageId} className="font-mono text-[10px] uppercase tracking-wider text-amber/70 text-center py-1 border-y border-dashed border-line">
              {msg.text}
            </p>
          ) : (
            <ChatLine
              key={msg.messageId}
              msg={msg}
              team={teamByPlayerId[msg.playerId]}
              isMine={msg.playerId === playerId}
            />
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
          className="field flex-1 min-w-0 !py-2 !text-sm"
        />
        <button
          type="submit"
          disabled={!isConnected || !text.trim()}
          className="relative bg-red hover:bg-red-glow text-bone font-display text-sm tracking-[0.08em] px-4 transition-colors
                     disabled:bg-raised disabled:text-bone/30 disabled:cursor-not-allowed"
          title={isConnected ? 'Connected' : 'Reconnecting…'}
        >
          SEND
          {/* Connection indicator — moved here from the old inner heading */}
          <span
            className={`absolute top-1 right-1 h-1.5 w-1.5 ${
              isConnected ? 'bg-cyan shadow-[0_0_6px_var(--color-cyan)]' : 'bg-amber animate-blink'
            }`}
          />
        </button>
      </form>
    </div>
  )
}

// One chat message. The sender's name is a tag in their team's colors
// (CSK → yellow, MI → blue…); players without a team fall back to plain
// red text. Your own tag gets a bone outline so you can spot your lines.
const ChatLine = ({ msg, team, isMine }) => (
  <div className="text-sm leading-snug">
    {team ? (
      <span
        className={`team-chip !normal-case !tracking-normal !text-[11px] mr-1.5 align-[1px]
          ${isMine ? 'outline outline-1 outline-offset-1 outline-bone/70' : ''}`}
        style={teamChipStyle(team)}
        title={team.name}
      >
        {msg.nickname}
      </span>
    ) : (
      <span className="font-mono text-[11px] text-red mr-1">{msg.nickname}&gt;</span>
    )}
    <span className="text-bone/90 break-words">{msg.text}</span>
  </div>
)

export default ChatPanel