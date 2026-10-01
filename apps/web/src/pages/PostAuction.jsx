import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { computeAwards, rankSquads } from '@ipl-auction/shared'
import { getAuctionResults } from '../lib/api'
import { findMyTeamId } from '../lib/results'
import { useLeaveRoom } from '../hooks/useLeaveRoom'
import { TEAMS_BY_ID, teamChipStyle } from '../constants/teams'
import AuctionSummary from '../components/post-auction/AuctionSummary'
import TeamLeaderboard from '../components/post-auction/TeamLeaderboard'
import TeamDetails from '../components/post-auction/TeamDetails'
import BestXI from '../components/post-auction/BestXI'
import TopPurchases from '../components/post-auction/TopPurchases'
import AuctionHistory from '../components/post-auction/AuctionHistory'
import AllSquads from '../components/post-auction/AllSquads'
import YourFinish from '../components/post-auction/YourFinish'
import ChampionBanner from '../components/post-auction/ChampionBanner'
import AuctionAwards from '../components/post-auction/AuctionAwards'
import ShareResultsButton from '../components/post-auction/ShareResultsButton'
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

  // Ranked here with the shared algorithm (also what the server saved), so
  // results saved before the Squad Score existed rank the same way.
  const teams = useMemo(() => (results ? rankSquads(results.teams) : []), [results])
  const myTeamId = useMemo(() => (results ? findMyTeamId(results.teams, roomId) : null), [results, roomId])
  // Presentation only — reads the ranking, never changes it.
  const awards = useMemo(() => computeAwards(teams), [teams])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-city flex flex-col items-center justify-center gap-4">
        <Spinner size={36} />
        <p className="font-display text-3xl uppercase tracking-wide text-bone/60">Finalizing results…</p>
        <p className="label-mono">Compiling ledger</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-city flex flex-col items-center justify-center px-4 gap-5">
        <p className="font-mono font-bold text-red tracking-[0.4em]">ERROR</p>
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

  // View Team opens on your team, or the winner for spectators.
  const activeTeamId = selectedTeamId ?? myTeamId ?? teams[0]?.teamId ?? null
  const selectedTeam = teams.find((t) => t.teamId === activeTeamId) || null
  const myTeam = teams.find((t) => t.teamId === myTeamId) || null

  return (
    <div className="relative min-h-screen bg-city px-4 py-8 sm:px-8 overflow-hidden">
      <div className="relative max-w-5xl mx-auto space-y-6">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <button
            type="button"
            onClick={leaveRoom}
            className="link-back"
          >
            ← Back to Home
          </button>
          <div className="flex items-center gap-3">
            <span className="label-mono">Room {roomId}</span>
            <ShareResultsButton roomId={roomId} />
          </div>
        </div>

        {/* 1. Auction complete + champion */}
        <div className="pb-6 border-b-2 border-bone">
          <div className="flex items-center gap-3 mb-2">
            <span className="h-2 w-2 bg-red" />
            <p className="label-mono text-bone/60">Auction complete // Final ledger</p>
          </div>
          <h1 className="font-display uppercase leading-[0.85] text-bone text-6xl sm:text-8xl">
            Auction <span className="text-red glow-red">Results</span>
          </h1>
        </div>

        <ChampionBanner champions={awards.champion?.teams} myTeamId={myTeamId} />

        {/* 2. Awards  3. Final standings  4. Your result */}
        <AuctionAwards awards={awards} myTeamId={myTeamId} num="01" />
        <TeamLeaderboard teams={teams} myTeamId={myTeamId} num="02" />
        <YourFinish team={myTeam} teamCount={teams.length} />

        {/* 5. All squads, then the single-team breakdown */}
        <AllSquads teams={teams} myTeamId={myTeamId} num="03" />

        <div>
          <div className="section-head">
            <span className="section-num">04</span>
            <h2 className="section-title">Team Breakdown</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            {teams.map((t) => (
              <button
                key={t.teamId}
                type="button"
                onClick={() => setSelectedTeamId(t.teamId)}
                title={t.teamId === myTeamId ? 'Your team' : undefined}
                className={`team-chip !text-xs !px-3 !py-2 transition-all
                  ${t.teamId === myTeamId ? 'ring-1 ring-cyan ring-offset-2 ring-offset-carbon' : ''}
                  ${activeTeamId === t.teamId
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
          <TeamDetails team={selectedTeam} num="05" />
          <BestXI team={selectedTeam} num="06" />
        </div>

        <AuctionSummary results={results} num="07" />
        <TopPurchases teams={teams} num="08" />

        {/* 6. Auction history */}
        <AuctionHistory history={results.history} num="09" />
      </div>
    </div>
  )
}

export default PostAuction