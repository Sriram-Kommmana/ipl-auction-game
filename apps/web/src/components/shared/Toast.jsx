import { useEffect } from 'react'

// Rendered ONCE, globally (see App.jsx) — fed directly by useSocket()'s
// socketError, rather than prop-drilled through every route like before.
const Toast = ({ message, onDismiss, duration = 4000 }) => {
  useEffect(() => {
    if (!message) return
    const timer = setTimeout(onDismiss, duration)
    return () => clearTimeout(timer)
  }, [message, duration, onDismiss])

  if (!message) return null

  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 px-4 w-full max-w-sm">
      <div className="bg-panel border-2 border-red shadow-brutal flex items-stretch">
        <div className="hazard w-2 shrink-0" />
        <div className="flex-1 px-4 py-3">
          <p className="font-mono text-[10px] uppercase tracking-[0.25em] text-red mb-0.5">Error // エラー</p>
          <p className="text-sm text-bone">{message}</p>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          className="px-3 text-bone/50 hover:text-red text-xl leading-none transition-colors"
          aria-label="Dismiss"
        >
          ×
        </button>
      </div>
    </div>
  )
}

export default Toast
