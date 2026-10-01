import { useState } from 'react'

// Copies this results page's link — same clipboard pattern as RoomCode.
// Results are public by room code, so the link works for anyone.
const ShareResultsButton = ({ roomId }) => {
  const [state, setState] = useState('idle') // idle | copied | failed

  const handleShare = async () => {
    try {
      await navigator.clipboard.writeText(`${window.location.origin}/results/${roomId}`)
      setState('copied')
    } catch {
      // Clipboard API can fail (permissions, non-secure context) — the
      // address bar still has the link.
      setState('failed')
    }
    setTimeout(() => setState('idle'), 1800)
  }

  return (
    <button
      type="button"
      onClick={handleShare}
      className={`inline-flex items-center gap-1.5 border px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] whitespace-nowrap transition-colors
        ${state === 'copied' ? 'border-cyan/60 text-cyan' : state === 'failed' ? 'border-red/60 text-red' : 'border-line-strong text-bone/60 hover:border-red hover:text-red'}`}
    >
      <span aria-hidden>{state === 'copied' ? '✓' : '⧉'}</span>
      <span aria-live="polite">{state === 'copied' ? 'Copied!' : state === 'failed' ? 'Copy failed' : 'Share Results'}</span>
    </button>
  )
}

export default ShareResultsButton
