import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { createRoom } from '../../lib/api'
import { saveSession } from '../../lib/session'
import { connectAndReconnect } from '../../lib/socket'
import { useSessionStore } from '../../store/sessionStore'
import Spinner from '../shared/Spinner'

const inputClass = 'field'

const labelClass = 'label-mono block mb-1.5'

const MODES = [
  { id: 'multiplayer', label: 'With Friends' },
  { id: 'solo', label: 'Solo vs AI' }
]

const POOLS = [
  { id: 'quick', label: 'Quick', detail: '140 players · 20–40 min' },
  { id: 'full', label: 'Full', detail: '323 players + re-auction' }
]

// Small segmented control in the terminal's tab style.
const Segmented = ({ options, value, onChange, label }) => (
  <div role="radiogroup" aria-label={label} className="grid grid-cols-2 border border-line">
    {options.map((opt, i) => (
      <button
        key={opt.id}
        type="button"
        role="radio"
        aria-checked={value === opt.id}
        data-active={value === opt.id}
        onClick={() => onChange(opt.id)}
        className={`tab py-2.5 px-2 text-base leading-tight ${i > 0 ? 'border-l border-line' : ''}`}
      >
        {opt.label}
        {opt.detail && (
          <span className="block font-mono text-[9px] tracking-[0.12em] normal-case opacity-70 mt-0.5">
            {opt.detail}
          </span>
        )}
      </button>
    ))}
  </div>
)

const CreateRoomForm = () => {
  const navigate = useNavigate()
  const setSession = useSessionStore((s) => s.setSession)

  const [managerNickname, setManagerNickname] = useState('')
  const [managerPin, setManagerPin] = useState('')
  const [roomPin, setRoomPin] = useState('')
  const [pursePerTeam, setPursePerTeam] = useState(12500)
  const [mode, setMode] = useState('multiplayer')
  const [pool, setPool] = useState('quick')
  const isSolo = mode === 'solo'
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
    if (!isSolo && !/^\d{4}$/.test(roomPin)) {
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
        roomPin: isSolo ? undefined : roomPin,
        pursePerTeam: Number(pursePerTeam),
        mode,
        pool: isSolo ? pool : 'full'
      })

      const { roomId, managerId, isManager } = res.data

      saveSession({
        uuid: managerId,
        roomId,
        nickname: managerNickname.trim(),
        playerPin: managerPin,
        roomPin: isSolo ? '' : roomPin,
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
        <span className={labelClass}>Opponents</span>
        <Segmented options={MODES} value={mode} onChange={setMode} label="Opponents" />
        {isSolo && (
          <p className="font-mono text-[10px] leading-relaxed text-bone/45 mt-2">
            You vs 9 AI franchises — 4 rule-based, 5 reinforcement-learning.
          </p>
        )}
      </div>

      {isSolo && (
        <div>
          <span className={labelClass}>Player Pool</span>
          <Segmented options={POOLS} value={pool} onChange={setPool} label="Player pool" />
        </div>
      )}

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

      <div className={`grid gap-3 ${isSolo ? 'grid-cols-1' : 'grid-cols-2'}`}>
        <div>
          <label className={labelClass}>Your PIN</label>
          <input
            type="password"
            inputMode="numeric"
            value={managerPin}
            onChange={(e) => setManagerPin(e.target.value.replace(/\D/g, '').slice(0, 4))}
            placeholder="4 digits"
            className={`${inputClass} tracking-[0.4em]`}
          />
        </div>
        {!isSolo && (
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
        )}
      </div>

      {/* <div>
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
      </div> */}

      {error && <p className="font-mono text-xs text-red border-l-2 border-red bg-red/10 px-3 py-2" role="alert">! {error}</p>}

      <button
        type="submit"
        disabled={isSubmitting}
        className="btn-primary w-full text-2xl py-3 flex items-center justify-center gap-3 !mt-6"
      >
        {isSubmitting && <Spinner size={20} variant="light" />}
        {isSubmitting ? 'CREATING…' : isSolo ? 'START SOLO GAME' : 'CREATE ROOM'}
      </button>
    </form>
  )
}

export default CreateRoomForm