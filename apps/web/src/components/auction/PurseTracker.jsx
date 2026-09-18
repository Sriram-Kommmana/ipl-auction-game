import { useMemo } from 'react'
import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'
import { useRoomStore } from '../../store/roomStore'
import { useSessionStore } from '../../store/sessionStore'

const PurseTracker = () => {
  const teams = useRoomStore((s) => s.teams)
  const players = useRoomStore((s) => s.players)
  const myTeamId = useSessionStore((s) => s.teamId)

  const nicknameByPlayerId = useMemo(
    () => Object.fromEntries(players.map((p) => [p.playerId, p.nickname])),
    [players]
  )

  const sorted = useMemo(() => {
    const claimed = teams.filter((t) => t.ownerId)
    return [...claimed].sort((a, b) => b.purseLeft - a.purseLeft)
  }, [teams])

  const topPurse = sorted[0]?.purseLeft

  if (sorted.length === 0) {
    return (
      <div className="panel p-4 h-full">
        <div className="section-head">
          <span className="section-num">01</span>
          <h2 className="section-title">Purse Tracker</h2>
          <span className="section-jp">残高</span>
        </div>
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-bone/30 text-center py-4">No teams have joined yet</p>
      </div>
    )
  }

  return (
    <div className="panel p-4 h-full flex flex-col">
      <div className="section-head">
        <span className="section-num">01</span>
        <h2 className="section-title">Purse Tracker</h2>
        <span className="font-mono text-[10px] text-red">[{String(sorted.length).padStart(2, '0')} ACTIVE]</span>
        <span className="section-jp">残高</span>
      </div>
      <div className="space-y-2 flex-1 min-h-0 overflow-y-auto pr-1">
        {sorted.map((team) => {
          const meta = TEAMS_BY_ID[team.teamId]
          const isMine = team.teamId === myTeamId
          const nickname = nicknameByPlayerId[team.ownerId]
          const isLeader = team.purseLeft === topPurse

          return (
            <div
              key={team.teamId}
              className={`row flex items-center justify-between px-3 py-2
                ${isMine ? '!border-cyan/60 !border-l-2 !border-l-cyan' : ''}`}
            >
              <div className="flex items-center gap-2 min-w-0">
                {isLeader && <span className="text-amber text-xs shrink-0" title="Deepest purse">▲</span>}
                <span className="team-chip shrink-0" style={teamChipStyle(meta)}>
                  {team.teamId}
                </span>
                <div className="min-w-0">
                  {nickname && (
                    <p className={`text-xs truncate leading-tight ${isMine ? 'text-cyan' : 'text-bone/80'}`}>{nickname}</p>
                  )}
                  <p className="font-mono text-[10px] text-bone/40 leading-tight mt-0.5">
                    {team.playerCount} PLR · {team.overseasCount} OS
                  </p>
                </div>
              </div>
              <span className={`num text-lg shrink-0 ${isLeader ? 'text-amber' : 'text-bone'}`}>₹{team.purseLeft}L</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default PurseTracker