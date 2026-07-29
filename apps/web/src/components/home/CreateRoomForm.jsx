import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { createRoom } from '../../lib/api'
import { saveSession } from '../../lib/session'
import { connectAndReconnect } from '../../lib/socket'
import { useSessionStore } from '../../store/sessionStore'

const inputClass =
  'w-full bg-mist border border-line rounded-lg px-3 py-2 text-ink ' +
  'placeholder:text-ink/30 focus:outline-none focus:ring-2 focus:ring-brand-red'

const labelClass = 'block text-xs uppercase tracking-wider text-ink/50 mb-1'

const CreateRoomForm = () => {
  const navigate = useNavigate()
  const setSession = useSessionStore((s) => s.setSession)

  const [managerNickname, setManagerNickname] = useState('')
  const [managerPin, setManagerPin] = useState('')
  const [roomPin, setRoomPin] = useState('')
  const [pursePerTeam, setPursePerTeam] = useState(12500)
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const validate = () => {
    const nickname = managerNickname.trim()
    if (nickname.length < 2 || nickname.length > 20) {
      return 'Nickname must be 2-20 characters.'
    }
    if (nickname.includes(':')) {
      return "Nickname cannot contain ':'."
    }
    if (!/^\d{4}$/.test(managerPin)) {
      return 'Your PIN must be exactly 4 digits.'
    }
    if (!/^\d{4}$/.test(roomPin)) {
      return 'Room PIN must be exactly 4 digits.'
    }
    if (pursePerTeam < 5000 || pursePerTeam > 50000) {
      return 'Purse per team must be between 5000 and 50000 lakhs.'
    }
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

    try {
      const res = await createRoom({
        managerNickname: managerNickname.trim(),
        managerPin,
        roomPin,
        pursePerTeam: Number(pursePerTeam)
      })

      const { roomId, managerId, isManager } = res.data

      saveSession({
        uuid: managerId,
        roomId,
        nickname: managerNickname.trim(),
        playerPin: managerPin,
        roomPin,
        isManager
      })

      setSession({
        playerId: managerId,
        roomId,
        teamId: '',
        nickname: managerNickname.trim(),
        isManager
      })

      connectAndReconnect(managerId)

      navigate(`/lobby/${roomId}`)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className={labelClass}>Your Name</label>
        <input
          type="text"
          value={managerNickname}
          onChange={(e) => setManagerNickname(e.target.value)}
          placeholder="e.g. Virat"
          maxLength={20}
          className={inputClass}
        />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className={labelClass}>Your PIN</label>
          <input
            type="password"
            inputMode="numeric"
            value={managerPin}
            onChange={(e) => setManagerPin(e.target.value.replace(/\D/g, '').slice(0, 4))}
            placeholder="4 digits"
            className={`${inputClass} tracking-widest`}
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
      </div>

      <div>
        <label className={labelClass}>Purse Per Team (lakhs)</label>
        <input
          type="number"
          min={5000}
          max={50000}
          step={500}
          value={pursePerTeam}
          onChange={(e) => setPursePerTeam(e.target.value)}
          className={inputClass}
        />
      </div>

      {error && <p className="text-sm text-brand-red-dark" role="alert">{error}</p>}

      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full bg-brand-red hover:bg-brand-red-dark disabled:opacity-50 disabled:cursor-not-allowed
                   text-paper font-display text-xl tracking-wide py-3 rounded-lg transition-colors"
      >
        {isSubmitting ? 'CREATING…' : 'CREATE ROOM'}
      </button>
    </form>
  )
}

export default CreateRoomForm