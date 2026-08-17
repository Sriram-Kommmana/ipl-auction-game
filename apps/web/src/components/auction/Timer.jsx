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
        className={`flex flex-col items-center justify-center rounded-lg border px-3 py-1.5 shrink-0
          ${isLow ? 'border-brand-red' : 'border-line'}`}
      >
        <span
          className={`font-display text-xl leading-none tabular-nums
            ${isLow ? 'text-brand-red' : 'text-ink'}`}
        >
          {secondsLeft}
        </span>
        <span className="text-[10px] text-ink/40 uppercase tracking-wide">
          {isPaused ? 'Paused' : 'Seconds'}
        </span>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center">
      <div
        className={`font-display text-5xl tabular-nums leading-none transition-colors
          ${isLow ? 'text-brand-red animate-pulse' : 'text-ink'}
        `}
      >
        {String(secondsLeft).padStart(2, '0')}
      </div>
      <p className="text-xs uppercase tracking-widest text-ink/40 mt-1">
        {isPaused ? 'Paused' : timerState === 'RUNNING' ? 'seconds left' : timerState}
      </p>
    </div>
  )
}

export default Timer