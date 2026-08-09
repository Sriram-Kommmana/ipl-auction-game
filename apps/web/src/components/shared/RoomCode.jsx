import { useState } from 'react'

const RoomCode = ({ roomId, size = 'lg' }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(roomId)
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
        {copied ? '✓ Copied' : '⧉'}
      </span>
    </button>
  )
}

export default RoomCode