import { useState } from 'react'

const RoomCode = ({ roomId, size = 'lg' }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      // Copies the full join link, not just the bare code — a link auto-
      // fills the Join form for whoever opens it, while a bare code still
      // requires typing it in manually. The code itself is still shown
      // visually below, for anyone who wants to read/type it by hand.
      const joinLink = `${window.location.origin}/join/${roomId}`
      await navigator.clipboard.writeText(joinLink)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard API can fail (permissions, non-secure context) — the
      // code is still visible on screen to copy manually either way.
    }
  }

  const isLarge = size === 'lg'

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={`group inline-flex items-center text-left border transition-colors
        ${isLarge
          ? 'gap-3 sm:gap-4 max-w-full border-line-strong bg-void pl-3 sm:pl-4 pr-3 py-2 hover:border-red'
          : 'gap-2 border-line bg-panel px-2.5 py-1.5 hover:border-red'}`}
      title="Click to copy"
    >
      <span
        className={`font-display text-bone leading-none
          ${isLarge ? 'text-4xl sm:text-5xl tracking-[0.12em] sm:tracking-[0.18em]' : 'text-base tracking-[0.18em]'}`}
      >
        {roomId}
      </span>
      <span
        className={`font-mono uppercase tracking-[0.15em] whitespace-nowrap transition-colors
          ${isLarge ? 'text-[10px]' : 'text-[9px]'}
          ${copied ? 'text-cyan' : 'text-bone/35 group-hover:text-red'}`}
      >
        {copied ? '✓' : '⧉'}
        <span className={isLarge ? 'ml-1' : 'hidden sm:inline ml-1'}>{copied ? 'Copied' : 'Copy Link'}</span>
      </span>
    </button>
  )
}

export default RoomCode
