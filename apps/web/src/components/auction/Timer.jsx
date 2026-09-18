// apps/web/src/components/auction/Timer.jsx
import { useTimer } from '../../hooks/useTimer'

// variant="full" — the big centered number (used elsewhere if needed)
// variant="compact" — small inline box, used next to BidButton
const Timer = ({ variant = 'full' }) => {
  const { secondsLeft, timerState } = useTimer()

  const isLow = secondsLeft <= 10 && timerState === 'RUNNING'
  const isPaused = timerState === 'PAUSED'

  if (variant === 'compact') {
    return (
      <div
        className={`h-full flex flex-col items-center justify-center border-2 px-3 py-1 shrink-0 transition-colors
          ${isLow ? 'border-red bg-red/15' : isPaused ? 'border-amber bg-amber/10' : 'border-line-strong bg-panel'}`}
      >
        <span
          className={`font-display text-3xl leading-none tabular-nums
            ${isLow ? 'text-red glow-red animate-pulse' : isPaused ? 'text-amber' : 'text-bone'}`}
        >
          {secondsLeft}
        </span>
        <span className={`font-mono text-[9px] uppercase tracking-[0.18em] mt-1 ${isPaused ? 'text-amber' : 'text-bone/40'}`}>
          {isPaused ? 'Paused' : 'Seconds'}
        </span>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center">
      <div
        className={`font-display text-5xl tabular-nums leading-none transition-colors
          ${isLow ? 'text-red glow-red animate-pulse' : 'text-bone'}
        `}
      >
        {String(secondsLeft).padStart(2, '0')}
      </div>
      <p className="label-mono mt-1">
        {isPaused ? 'Paused' : timerState === 'RUNNING' ? 'seconds left' : timerState}
      </p>
    </div>
  )
}

export default Timer