// TEMPORARY placeholder — real PostAuction UI comes next in the build order.
import { useParams } from 'react-router-dom'
import { useRoomStore } from '../store/roomStore'

const PostAuction = () => {
  const { roomId } = useParams()
  const teams = useRoomStore((s) => s.teams)

  return (
    <div className="min-h-screen bg-mist p-8">
      <h1 className="font-display text-4xl text-ink">RESULTS — {roomId}</h1>
      <pre className="text-xs mt-4 bg-paper p-4 rounded-lg border border-line overflow-auto">
        {JSON.stringify(teams, null, 2)}
      </pre>
    </div>
  )
}

export default PostAuction