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
      <div className="bg-brand-red text-paper rounded-xl px-4 py-3 shadow-lg flex items-center justify-between gap-3">
        <p className="text-sm">{message}</p>
        <button
          type="button"
          onClick={onDismiss}
          className="text-paper/70 hover:text-paper text-lg leading-none"
          aria-label="Dismiss"
        >
          ×
        </button>
      </div>
    </div>
  )
}

export default Toast