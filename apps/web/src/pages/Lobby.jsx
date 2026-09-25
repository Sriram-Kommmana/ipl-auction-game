import { useParams } from 'react-router-dom'
import { useRoomStore } from '../store/roomStore'
import TeamGrid from '../components/lobby/TeamGrid'
import PlayerList from '../components/lobby/PlayerList'
import LobbyControls from '../components/lobby/LobbyControls'
import ChatPanel from '../components/shared/ChatPanel'
import RoomCode from '../components/shared/RoomCode'
import OpponentsPanel from '../components/lobby/OpponentsPanel'

const Lobby = () => {
  const { roomId } = useParams()
  const pursePerTeam = useRoomStore((s) => s.pursePerTeam)
  const isSolo = useRoomStore((s) => s.mode === 'solo')

  return (
    <div className="relative min-h-screen bg-city px-4 py-8 sm:px-8">
      {/* Watermark clips itself — the page wrapper must NOT be
          overflow-hidden, or the sticky action bar below can't stick. */}
      <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
        <div
          className="hidden sm:block select-none absolute -right-4 top-10 font-jp font-black
                     text-[8rem] sm:text-[12rem] leading-none text-bone/[0.025]"
        >
          待機
        </div>
      </div>

      <div className="relative max-w-5xl mx-auto">
        {/* Masthead */}
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-6 pb-6 mb-8 border-b-2 border-bone">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="h-2 w-2 bg-red animate-blink" />
              <p className="label-mono text-bone/60">
                {isSolo ? 'Solo // 単独戦 // Pick your franchise' : 'Lobby // 待機室 // Awaiting operators'}
              </p>
            </div>
            <h1 className="font-display text-6xl sm:text-7xl uppercase leading-[0.85] text-bone">
              The <span className="text-red glow-red">Lobby</span>
            </h1>
          </div>

          <div className="flex items-end gap-6">
            {isSolo ? (
              <div>
                <p className="label-mono mb-1.5">Mode</p>
                <p className="num text-3xl text-bone leading-none">
                  Solo <span className="text-red">vs</span> 9 AI
                </p>
              </div>
            ) : (
              <div>
                <p className="label-mono mb-1.5">Room Code</p>
                <RoomCode roomId={roomId} size="lg" />
              </div>
            )}
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

        {/* Teams + Players stack in the left column so the right column is
            free for chat — that keeps the page short enough that Start
            Auction / Leave Room stay reachable without a long scroll. */}
        <div className="grid md:grid-cols-3 gap-6 items-stretch">
          <div className="md:col-span-2 flex flex-col gap-6">
            <TeamGrid />
            {!isSolo && <PlayerList />}
          </div>

          {/* Solo: nobody to chat with, so show who you're up against.
              Multiplayer: the same ChatPanel the auction uses — chat is
              room-wide, so the backend accepts messages while the room is
              still waiting and the history carries straight over. */}
          {isSolo ? (
            <OpponentsPanel />
          ) : (
            <div className="panel p-4 flex flex-col h-96 md:h-full">
              <div className="section-head">
                <span className="section-num">03</span>
                <h2 className="section-title">Chat</h2>
                <span className="section-jp">通信</span>
              </div>
              <div className="flex-1 min-h-0">
                <ChatPanel />
              </div>
            </div>
          )}
        </div>

        <LobbyControls />
      </div>
    </div>
  )
}

export default Lobby
