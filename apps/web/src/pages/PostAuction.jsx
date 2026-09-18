import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { getAuctionResults } from '../lib/api'
import { useLeaveRoom } from '../hooks/useLeaveRoom'
import { TEAMS_BY_ID, teamChipStyle } from '../constants/teams'
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
      <div className="min-h-screen bg-city flex flex-col items-center justify-center gap-4">
        <Spinner size={36} />
        <p className="font-display text-3xl uppercase tracking-wide text-bone/60">Finalizing results…</p>
        <p className="label-mono">集計中 // Compiling ledger</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-city flex flex-col items-center justify-center px-4 gap-5">
        <p className="font-jp font-black text-red tracking-[0.4em]">エラー</p>
        <p className="font-mono text-sm text-bone/80 text-center border-l-2 border-red bg-red/10 px-4 py-3 max-w-md">
          ! {error}
        </p>
        <button
          type="button"
          onClick={leaveRoom}
          className="link-back"
        >
          ← Back to Home
        </button>
      </div>
    )
  }

  const selectedTeam = results.teams.find((t) => t.teamId === selectedTeamId) || null

  return (
    <div className="relative min-h-screen bg-city px-4 py-8 sm:px-8 overflow-hidden">
      <div
        aria-hidden
        className="pointer-events-none select-none absolute -right-6 top-72 sm:top-6 font-jp font-black
                   text-[8rem] sm:text-[13rem] leading-none text-bone/[0.025]"
      >
        結果
      </div>

      <div className="relative max-w-5xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={leaveRoom}
            className="link-back"
          >
            ← Back to Home
          </button>
          <span className="label-mono">Room {roomId}</span>
        </div>

        <div className="pb-6 border-b-2 border-bone">
          <div className="flex items-center gap-3 mb-2">
            <span className="h-2 w-2 bg-red" />
            <p className="label-mono text-bone/60">Hammer down // 終了 // Final ledger</p>
          </div>
          <h1 className="font-display uppercase leading-[0.85] text-bone text-6xl sm:text-8xl">
            Auction <span className="text-red glow-red">Results</span>
          </h1>
        </div>

        <AuctionSummary results={results} />
        <TeamLeaderboard teams={results.teams} />
        <TopPurchases teams={results.teams} />

        <div>
          <div className="section-head">
            <span className="section-num">04</span>
            <h2 className="section-title">View Team</h2>
            <span className="section-jp">チーム詳細</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {results.teams.map((t) => (
              <button
                key={t.teamId}
                type="button"
                onClick={() => setSelectedTeamId(t.teamId)}
                className={`team-chip !text-xs !px-3 !py-2 transition-all
                  ${selectedTeamId === t.teamId
                    ? 'outline-2 outline-offset-2 outline-bone -translate-y-0.5'
                    : 'opacity-50 hover:opacity-100'}`}
                style={teamChipStyle(TEAMS_BY_ID[t.teamId])}
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