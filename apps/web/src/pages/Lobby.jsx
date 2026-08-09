import { useParams } from 'react-router-dom'
import { useRoomStore } from '../store/roomStore'
import TeamGrid from '../components/lobby/TeamGrid'
import PlayerList from '../components/lobby/PlayerList'
import LobbyControls from '../components/lobby/LobbyControls'
import RoomCode from '../components/shared/RoomCode'

const Lobby = () => {
  const { roomId } = useParams()
  const pursePerTeam = useRoomStore((s) => s.pursePerTeam)

  return (
    <div className="min-h-screen bg-mist px-4 py-8 sm:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <p className="text-xs uppercase tracking-wider text-ink/50">Room Code</p>
          <RoomCode roomId={roomId} size="lg" />
          <p className="text-xs text-ink/40 mt-1">Purse per team: ₹{pursePerTeam}L</p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="md:col-span-2">
            <TeamGrid />
          </div>
          <div>
            <PlayerList />
          </div>
        </div>

        <LobbyControls />
      </div>
    </div>
  )
}

export default Lobby