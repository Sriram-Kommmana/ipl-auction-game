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
    <div className="relative min-h-screen bg-city px-4 py-8 sm:px-8 overflow-hidden">
      <div
        aria-hidden
        className="hidden sm:block pointer-events-none select-none absolute -right-4 top-10 font-jp font-black
                   text-[8rem] sm:text-[12rem] leading-none text-bone/[0.025]"
      >
        待機
      </div>

      <div className="relative max-w-5xl mx-auto">
        {/* Masthead */}
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-6 pb-6 mb-8 border-b-2 border-bone">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="h-2 w-2 bg-red animate-blink" />
              <p className="label-mono text-bone/60">Lobby // 待機室 // Awaiting operators</p>
            </div>
            <h1 className="font-display text-6xl sm:text-7xl uppercase leading-[0.85] text-bone">
              The <span className="text-red glow-red">Lobby</span>
            </h1>
          </div>

          <div className="flex items-end gap-6">
            <div>
              <p className="label-mono mb-1.5">Room Code</p>
              <RoomCode roomId={roomId} size="lg" />
            </div>
            <div className="hidden sm:block text-right">
              <p className="label-mono mb-1.5">Purse / Team</p>
              <p className="num text-3xl text-bone leading-none">
                ₹{pursePerTeam}<span className="text-red">L</span>
              </p>
            </div>
          </div>
        </div>
        <p className="sm:hidden label-mono -mt-5 mb-6">
          Purse per team: <span className="text-bone">₹{pursePerTeam}L</span>
        </p>

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
