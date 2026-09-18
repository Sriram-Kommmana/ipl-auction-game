import { ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { joinRoom } from '../../lib/api'
import { saveSession, getSession } from '../../lib/session'
import { connectAndReconnect } from '../../lib/socket'
import { getRoomRoute } from '../../lib/routing'
import { useSessionStore } from '../../store/sessionStore'
import { useRejoinRecent } from '../../hooks/useRejoinRecent'
import Spinner from '../shared/Spinner'

const inputClass = 'field'

const labelClass = 'label-mono block mb-1.5'

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
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className={labelClass}>Room Code</label>
          <input
            type="text"
            value={roomId}
            onChange={(e) => setRoomId(e.target.value.toUpperCase().slice(0, 6))}
            placeholder="e.g. X7K2AB"
            className={`${inputClass} tracking-[0.3em] uppercase`}
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
            className={`${inputClass} tracking-[0.4em]`}
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
            className={`${inputClass} tracking-[0.4em]`}
          />
        </div>

        {error && <p className="font-mono text-xs text-red border-l-2 border-red bg-red/10 px-3 py-2" role="alert">! {error}</p>}

        <button
          type="submit"
          disabled={isSubmitting}
          className="btn-primary w-full text-2xl py-3 flex items-center justify-center gap-3 !mt-6"
        >
          {isSubmitting && <Spinner size={20} variant="light" />}
          {isSubmitting ? 'JOINING…' : 'JOIN ROOM'}
        </button>
      </form>

      {recents.length > 0 && (
        <div className="mt-8 pt-6 border-t border-dashed border-line-strong">
          <div className="flex items-baseline justify-between mb-3">
            <p className="label-mono text-bone/70">Continue Playing</p>
            <p className="font-jp text-[10px] tracking-[0.3em] text-bone/25">続行</p>
          </div>
          <div className="space-y-2">
            {recents.map((recent) => (
              <button
                key={recent.roomId}
                type="button"
                onClick={() => rejoinRecent(recent)}
                className="w-full flex items-center justify-between bg-void border border-line
                           border-l-[3px] border-l-red px-4 py-3 text-left
                           hover:bg-raised hover:border-line-strong hover:border-l-red transition-colors group"
              >
                <div>
                  <p className="font-display text-2xl tracking-[0.15em] text-bone leading-none">
                    {recent.roomId}
                  </p>
                  <p className="font-mono text-[11px] text-bone/45 mt-1.5 uppercase tracking-wider">
                    {recent.nickname}
                    {recent.isManager && <span className="text-amber"> · Manager</span>}
                  </p>
                </div>
                <ChevronRight
                  size={22}
                  className="text-bone/30 group-hover:text-red group-hover:translate-x-1 transition-all"
                />
              </button>
            ))}
          </div>
          <p className="text-xs text-bone/40 mt-3 leading-relaxed">
            Returning player? Tap a room above to rejoin instantly. New here? Use the form above.
          </p>
        </div>
      )}
    </div>
  )
}

export default JoinRoomForm