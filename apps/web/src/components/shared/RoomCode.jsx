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

  const textSize = size === 'lg' ? 'text-4xl' : 'text-sm'

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="group inline-flex items-center gap-2 text-left"
      title="Click to copy"
    >
      <span className={`font-display ${textSize} text-ink tracking-widest`}>{roomId}</span>
      <span className="text-xs text-ink/30 group-hover:text-ink/60 transition-colors">
        {copied ? '✓ Link Copied' : '⧉ Copy Link'}
      </span>
    </button>
  )
}

export default RoomCode