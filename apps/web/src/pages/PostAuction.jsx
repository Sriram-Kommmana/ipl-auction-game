// apps/web/src/pages/PostAuction.jsx
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { getAuctionResults } from '../lib/api'
import { useLeaveRoom } from '../hooks/useLeaveRoom'
import { TEAMS_BY_ID } from '../constants/teams'
import AuctionSummary from '../components/post-auction/AuctionSummary'
import TeamLeaderboard from '../components/post-auction/TeamLeaderboard'
import TeamDetails from '../components/post-auction/TeamDetails'
import BestXI from '../components/post-auction/BestXI'
import TopPurchases from '../components/post-auction/TopPurchases'
import AuctionHistory from '../components/post-auction/AuctionHistory'
import Spinner from '../components/shared/Spinner'

const MAX_RETRIES = 5
const RETRY_DELAY_MS = 1500

const PostAuction = () => {
  const { roomId } = useParams()
  const leaveRoom = useLeaveRoom()
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [selectedTeamId, setSelectedTeamId] = useState(null)

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

  useEffect(() => {
    if (results && !selectedTeamId) {
      setSelectedTeamId(results.teams[0]?.teamId ?? null)
    }
  }, [results, selectedTeamId])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-mist flex items-center justify-center gap-4">
        <Spinner size={32} />
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

  const selectedTeam = results.teams.find((t) => t.teamId === selectedTeamId) || null

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

        <AuctionSummary results={results} />
        <TeamLeaderboard teams={results.teams} />
        <TopPurchases teams={results.teams} />

        <div>
          <p className="text-xs uppercase tracking-wide text-ink/50 mb-2">View Team</p>
          <div className="flex flex-wrap gap-2">
            {results.teams.map((t) => (
              <button
                key={t.teamId}
                type="button"
                onClick={() => setSelectedTeamId(t.teamId)}
                className={`text-xs font-display px-3 py-1.5 rounded-full text-white transition-all
                  ${selectedTeamId === t.teamId ? 'ring-2 ring-ink ring-offset-2 ring-offset-mist' : 'opacity-70'}`}
                style={{ backgroundColor: TEAMS_BY_ID[t.teamId]?.color }}
              >
                {t.teamId}
              </button>
            ))}
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <TeamDetails team={selectedTeam} />
          <BestXI team={selectedTeam} />
        </div>

        <AuctionHistory history={results.history} />
      </div>
    </div>
  )
}

export default PostAuction