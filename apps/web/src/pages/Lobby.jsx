// TEMPORARY placeholder — real Lobby UI (TeamGrid, PlayerList, LobbyControls)
// comes next in the build order. This just proves data flows correctly.
import { useParams } from 'react-router-dom'
import { useRoomStore } from '../store/roomStore'

const Lobby = () => {
  const { roomId } = useParams()
  const roomStatus = useRoomStore((s) => s.roomStatus)
  const players = useRoomStore((s) => s.players)
  const teams = useRoomStore((s) => s.teams)

  return (
    <div className="min-h-screen bg-mist p-8">
      <h1 className="font-display text-4xl text-ink">LOBBY — {roomId}</h1>
      <p className="text-ink/60 mt-1">status: {roomStatus}</p>
      <h2 className="font-display text-xl mt-6">Players</h2>
      <pre className="text-xs mt-2 bg-paper p-4 rounded-lg border border-line overflow-auto">
        {JSON.stringify(players, null, 2)}
      </pre>
      <h2 className="font-display text-xl mt-6">Teams</h2>
      <pre className="text-xs mt-2 bg-paper p-4 rounded-lg border border-line overflow-auto">
        {JSON.stringify(teams, null, 2)}
      </pre>
    </div>
  )
}

export default Lobby