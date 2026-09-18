import { useMemo } from 'react'
import { useRoomStore } from '../../store/roomStore'
import { useSessionStore } from '../../store/sessionStore'

const SquadViewer = () => {
  const history = useRoomStore((s) => s.history)
  const myTeamId = useSessionStore((s) => s.teamId)

  const mySquad = useMemo(() => {
    return [...history]
      .filter((h) => h.status === 'sold' && h.soldTo === myTeamId)
      .sort((a, b) => a.soldAt - b.soldAt)
  }, [history, myTeamId])

  if (!myTeamId) return null

  return (
    <div>

      {mySquad.length === 0 ? (
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-bone/30 text-center py-4">No players bought yet</p>
      ) : (
        <div className="space-y-1.5">
          {mySquad.map((entry) => (
            <div key={entry.iplPlayerId} className="row flex items-center justify-between text-sm px-2.5 py-1.5">
              <div className="flex items-center gap-1.5 min-w-0">
                <span className="text-bone truncate">{entry.playerName}</span>
                {entry.role && (
                  <span className="font-mono text-[9px] uppercase tracking-wider border border-line-strong text-bone/50 px-1.5 py-0.5 shrink-0 min-w-12 text-center">
                    {entry.role}
                  </span>
                )}
              </div>
              <span className="num text-lg text-bone shrink-0">₹{entry.soldFor}L</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default SquadViewer