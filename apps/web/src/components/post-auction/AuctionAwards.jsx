import { TEAMS_BY_ID, teamChipStyle } from '../../constants/teams'

// Award cards from computeAwards (packages/shared/src/auctionAwards.js —
// the formulas are documented there). Presentation only: nothing here
// feeds back into the Squad Score or the standings.

const YouTag = () => (
  <span className="font-mono text-[9px] tracking-[0.2em] text-cyan border border-cyan/60 px-1 py-0.5 shrink-0">YOU</span>
)

const TeamLine = ({ team, myTeamId }) => (
  <div className="flex items-center gap-2 min-w-0 mt-2">
    <span className="team-chip shrink-0" style={teamChipStyle(TEAMS_BY_ID[team.teamId])}>{team.teamId}</span>
    <span className="font-mono text-[11px] text-bone/55 truncate">{team.ownerNickname}</span>
    {team.teamId === myTeamId && <YouTag />}
  </div>
)

const AwardCard = ({ icon, title, hint, headline, team, myTeamId, stats, highlight }) => (
  <div
    className={`row !bg-raised/70 p-4 min-w-0 flex flex-col ${highlight ? '!border-amber/40' : '!border-line'}`}
    title={hint}
  >
    <p className={`label-mono flex items-center gap-2 ${highlight ? 'text-amber' : 'text-red'}`}>
      <span aria-hidden className="text-sm leading-none">{icon}</span>
      {title}
    </p>
    <p className="font-display text-xl sm:text-2xl uppercase tracking-wide text-bone leading-tight mt-2 truncate">{headline}</p>
    {team && <TeamLine team={team} myTeamId={myTeamId} />}
    <p className="font-mono text-[10px] text-bone/50 mt-auto pt-3 flex flex-wrap gap-x-3 gap-y-1">
      {stats.map(([label, value]) => (
        <span key={label} className="whitespace-nowrap">{label} <span className="text-bone/85">{value}</span></span>
      ))}
    </p>
  </div>
)

const AuctionAwards = ({ awards, myTeamId, num = '01' }) => {
  const cards = []
  const { champion, bestValue, biggestSplurge, bargain, deepestSquad, highestRated } = awards

  if (champion?.teams.length) {
    const lead = champion.teams[0]
    cards.push({
      key: 'champion', icon: '🏆', title: champion.teams.length > 1 ? 'Joint Champions' : 'Auction Champion', highlight: true,
      hint: 'Highest final Squad Score',
      headline: champion.teams.map((t) => t.teamName).join(' & '),
      team: champion.teams.length === 1 ? lead : null,
      stats: [['SQUAD SCORE', lead.ranking.score], ['XI', lead.ranking.xiStrength]]
    })
  }
  if (bestValue) {
    cards.push({
      key: 'bestValue', icon: '💎', title: 'Best Value Buy',
      hint: 'Biggest saving against the game\'s fair-value price for the player',
      headline: bestValue.player.playerName, team: bestValue.team,
      stats: [['PAID', `₹${bestValue.price}L`], ['WORTH', `₹${bestValue.fair}L`], ['SAVED', `₹${bestValue.saving}L`], ['★', bestValue.player.rating]]
    })
  }
  if (biggestSplurge) {
    cards.push({
      key: 'splurge', icon: '🔥', title: 'Biggest Splurge',
      hint: 'Highest price paid for one player',
      headline: biggestSplurge.player.playerName, team: biggestSplurge.team,
      stats: [['PAID', `₹${biggestSplurge.price}L`], ['★', biggestSplurge.player.rating]]
    })
  }
  if (bargain) {
    cards.push({
      key: 'bargain', icon: '💰', title: 'Bargain of the Auction',
      hint: 'Cheapest purchase of a player rated 88 or higher (other than the Best Value Buy)',
      headline: bargain.player.playerName, team: bargain.team,
      stats: [['★', bargain.player.rating], ['PAID', `₹${bargain.price}L`]]
    })
  }
  if (deepestSquad) {
    const t = deepestSquad.team
    cards.push({
      key: 'deepest', icon: '🧱', title: 'Deepest Squad',
      hint: 'Highest injury cover: expected XI strength with any one starter missing',
      headline: t.teamName, team: t,
      stats: [['INJURY COVER', t.ranking.injuryCover], ['SQUAD', `${t.playerCount} PLR`]]
    })
  }
  if (highestRated) {
    cards.push({
      key: 'highest', icon: '⭐', title: 'Highest Rated Player',
      hint: 'Highest-rated player bought in the auction',
      headline: highestRated.player.playerName, team: highestRated.team,
      stats: [['★', highestRated.player.rating], ['PAID', `₹${highestRated.price}L`]]
    })
  }

  if (!cards.length) return null

  return (
    <div className="panel p-4 sm:p-6">
      <div className="section-head">
        <span className="section-num">{num}</span>
        <h2 className="section-title">Auction Awards</h2>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {cards.map(({ key, ...card }) => <AwardCard key={key} {...card} myTeamId={myTeamId} />)}
      </div>
    </div>
  )
}

export default AuctionAwards
