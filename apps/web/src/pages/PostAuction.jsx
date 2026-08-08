// apps/web/src/pages/PostAuction.jsx
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { getAuctionResults } from '../lib/api'
import { useLeaveRoom } from '../hooks/useLeaveRoom'

const MAX_RETRIES = 5
const RETRY_DELAY_MS = 1500

const PostAuction = () => {
  const { roomId } = useParams()
  const leaveRoom = useLeaveRoom()
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let attempt = 0
    let cancelled = false
    let timeoutId = null

    const fetchResults = async () => {
      try {
        const res = await getAuctionResults(roomId)
        if (!cancelled) {
          setResults(res.data)
          setIsLoading(false)
        }
      } catch (err) {
        attempt += 1
        if (attempt < MAX_RETRIES) {
          // persistAuctionResults is fire-and-forget after auctionCompleted
          // — there's a real race where we can arrive here before MongoDB
          // has finished writing. Retry a few times before treating it as
          // an actual failure.
          timeoutId = setTimeout(fetchResults, RETRY_DELAY_MS)
        } else if (!cancelled) {
          setError(err.message)
          setIsLoading(false)
        }
      }
    }

    fetchResults()

    return () => {
      cancelled = true
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [roomId])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-mist flex items-center justify-center">
        <p className="font-display text-2xl text-ink/40">Finalizing results…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-mist flex flex-col items-center justify-center px-4 gap-4">
        <p className="text-brand-red-dark text-center">{error}</p>
        <button
          type="button"
          onClick={leaveRoom}
          className="text-sm text-ink/50 hover:text-brand-red transition-colors"
        >
          ← Back to Home
        </button>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-mist px-4 py-8 sm:px-8">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={leaveRoom}
            className="text-sm text-ink/50 hover:text-brand-red transition-colors"
          >
            ← Back to Home
          </button>
        </div>

        <h1 className="font-display text-4xl text-ink text-center tracking-wide">
          AUCTION RESULTS
        </h1>

        {/* TEMPORARY — replaced by the 6 real components, built next:
            <AuctionSummary results={results} />
            <TeamLeaderboard teams={results.teams} />
            <TeamDetails teams={results.teams} />
            <BestXI teams={results.teams} />
            <TopPurchases teams={results.teams} />
            <AuctionHistory history={results.history} />
        */}
        <pre className="text-xs bg-paper p-4 rounded-lg border border-line overflow-auto">
          {JSON.stringify(results, null, 2)}
        </pre>
      </div>
    </div>
  )
}

export default PostAuction