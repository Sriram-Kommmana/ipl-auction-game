import { ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { joinRoom } from '../../lib/api'
import { saveSession, getSession } from '../../lib/session'
import { connectAndReconnect } from '../../lib/socket'
import { getRoomRoute } from '../../lib/routing'
import { useSessionStore } from '../../store/sessionStore'
import { useRejoinRecent } from '../../hooks/useRejoinRecent'

const inputClass =
  'w-full bg-mist border border-line rounded-lg px-3 py-2 text-ink ' +
  'placeholder:text-ink/30 focus:outline-none focus:ring-2 focus:ring-brand-red'

const labelClass = 'block text-xs uppercase tracking-wider text-ink/50 mb-1'

const JoinRoomForm = ({ initialRoomId = '' }) => {
  const navigate = useNavigate()
  const setSession = useSessionStore((s) => s.setSession)
  const rejoinRecent = useRejoinRecent()

  const [roomId, setRoomId] = useState(initialRoomId.toUpperCase())
  const [roomPin, setRoomPin] = useState('')
  const [nickname, setNickname] = useState('')
  const [playerPin, setPlayerPin] = useState('')
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const recents = getSession()?.recents || []

  const validate = () => {
    if (!roomId.trim()) return 'Room code is required.'
    if (!/^\d{4}$/.test(roomPin)) return 'Room PIN must be exactly 4 digits.'
    const trimmed = nickname.trim()
    if (trimmed.length < 2 || trimmed.length > 20) return 'Nickname must be 2-20 characters.'
    if (trimmed.includes(':')) return "Nickname cannot contain ':'."
    if (!/^\d{4}$/.test(playerPin)) return 'Your PIN must be exactly 4 digits.'
    return ''
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    const validationError = validate()
    if (validationError) {
      setError(validationError)
      return
    }

    setError('')
    setIsSubmitting(true)

    const cleanRoomId = roomId.trim().toUpperCase()

    try {
      const res = await joinRoom({
        roomId: cleanRoomId,
        roomPin,
        nickname: nickname.trim(),
        playerPin
      })

      // roomStatus tells us if this is a rejoin into an already-active
      // auction (e.g. after Leave Room mid-auction) — previously this
      // always navigated to /lobby regardless, stranding rejoining
      // players who should have landed on /auction instead.
      const { playerId, teamId, isManager, roomStatus } = res.data

      saveSession({
        uuid: playerId,
        roomId: cleanRoomId,
        nickname: nickname.trim(),
        playerPin,
        roomPin,
        isManager
      })

      setSession({
        playerId,
        roomId: cleanRoomId,
        teamId: teamId || '',
        nickname: nickname.trim(),
        isManager
      })

      connectAndReconnect(playerId)

      navigate(getRoomRoute(roomStatus, cleanRoomId))
    } catch (err) {
      setError(err.message)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div>
      {recents.length > 0 && (
        <div className="mb-6">
          <p className="text-xs uppercase tracking-wider text-ink/50 mb-2">Continue Playing</p>
          <div className="space-y-2">
            {recents.map((recent) => (
              <button
                key={recent.roomId}
                type="button"
                onClick={() => rejoinRecent(recent)}
                className="w-full flex items-center justify-between bg-paper border border-line
                           rounded-xl px-4 py-3 text-left hover:border-brand-red transition-colors group"
              >
                <div>
                  <p className="font-display text-2xl tracking-wide text-ink leading-none">
                    {recent.roomId}
                  </p>
                  <p className="text-xs text-ink/50 mt-1">
                    {recent.nickname}{recent.isManager ? ' · Manager' : ''}
                  </p>
                </div>
                <ChevronRight
                  size={20}
                  className="text-ink/30 group-hover:text-brand-red transition-colors"
                />
              </button>
            ))}
          </div>
          <p className="text-xs text-ink/40 mt-2">
            Tap to rejoin instantly. New device? Use the form below.
          </p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className={labelClass}>Room Code</label>
          <input
            type="text"
            value={roomId}
            onChange={(e) => setRoomId(e.target.value.toUpperCase().slice(0, 6))}
            placeholder="e.g. X7K2AB"
            className={`${inputClass} tracking-widest uppercase`}
          />
        </div>

        <div>
          <label className={labelClass}>Room PIN</label>
          <input
            type="password"
            inputMode="numeric"
            value={roomPin}
            onChange={(e) => setRoomPin(e.target.value.replace(/\D/g, '').slice(0, 4))}
            placeholder="4 digits"
            className={`${inputClass} tracking-widest`}
          />
        </div>

        <div>
          <label className={labelClass}>Your Name</label>
          <input
            type="text"
            value={nickname}
            onChange={(e) => setNickname(e.target.value)}
            placeholder="e.g. Rahul"
            maxLength={20}
            className={inputClass}
          />
        </div>

        <div>
          <label className={labelClass}>Your PIN</label>
          <input
            type="password"
            inputMode="numeric"
            value={playerPin}
            onChange={(e) => setPlayerPin(e.target.value.replace(/\D/g, '').slice(0, 4))}
            placeholder="4 digits"
            className={`${inputClass} tracking-widest`}
          />
        </div>

        {error && <p className="text-sm text-brand-red-dark" role="alert">{error}</p>}

        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full bg-brand-red hover:bg-brand-red-dark disabled:opacity-50 disabled:cursor-not-allowed
                     text-paper font-display text-xl tracking-wide py-3 rounded-lg transition-colors"
        >
          {isSubmitting ? 'JOINING…' : 'JOIN ROOM'}
        </button>
      </form>
    </div>
  )
}

export default JoinRoomForm