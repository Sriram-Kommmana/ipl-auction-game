import { useTimer } from '../../hooks/useTimer'

const Timer = () => {
  const { secondsLeft, timerState } = useTimer()

  const isLow = secondsLeft <= 10 && timerState === 'RUNNING'
  const isPaused = timerState === 'PAUSED'

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