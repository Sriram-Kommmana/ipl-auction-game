// apps/web/src/components/auction/SquadViewer.jsx
import { useMemo } from 'react'
import { useRoomStore } from '../../store/roomStore'
import { useSessionStore } from '../../store/sessionStore'

const SquadViewer = () => {
  const history = useRoomStore((s) => s.history)
  const myTeamId = useSessionStore((s) => s.teamId)

  const mySquad = useMemo(
    () =>
      history
        .filter((h) => h.status === 'sold' && h.soldTo === myTeamId)
        .sort((a, b) => a.soldAt - b.soldAt),
    [history, myTeamId]
  )

  if (!myTeamId) return null

  return (
    <div className="bg-paper border border-line rounded-2xl p-4">
      <div className="flex items-center justify-between mb-2">
        <h2 className="font-display text-lg text-ink tracking-wide">MY SQUAD</h2>
        <span className="text-xs text-ink/40">{mySquad.length} players</span>
      </div>

      {mySquad.length === 0 ? (
        <p className="text-xs text-ink/30 text-center py-4">No players bought yet</p>
      ) : (
        <div className="space-y-1.5 max-h-72 overflow-y-auto">
          {mySquad.map((entry) => (
            <div key={entry.iplPlayerId} className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-1.5 min-w-0">
                <span className="text-ink truncate">{entry.playerName}</span>
                {entry.role && (
                  <span className="text-[10px] font-display bg-mist text-ink/50 px-1.5 py-0.5 rounded shrink-0">
                    {entry.role}
                  </span>
                )}
              </div>
              <span className="font-display text-ink shrink-0">₹{entry.soldFor}L</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default SquadViewer